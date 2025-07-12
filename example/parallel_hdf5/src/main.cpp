#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <filesystem>
#include <vector>

struct ClusterData {
    // Stellar mass (in solar masses)
    double stellar_mass;
    // Gas mass (in solar masses)
    double gas_mass;
    // Mass of DM contained within either virial radius (if virialized)
    // or FOF output radius (if not virialized) (in solar masses)
    double dark_matter_mass;
    // Half mass radius of object (in kpc)
    double half_mass_rad;
    // if in virial equilibrium (Ek balances Ug)
    // FIXME: dataset stores as float :,) when it should be boolean
    double virialized;

    static hid_t H5Type() {
        const hid_t ty = H5Tcreate(H5T_COMPOUND, sizeof(ClusterData));
        H5Tinsert(ty, "stellarMass (Msun)", offsetof(ClusterData, stellar_mass), H5T_NATIVE_DOUBLE);
        H5Tinsert(ty, "gasMass (Msun)", offsetof(ClusterData, gas_mass), H5T_NATIVE_DOUBLE);
        H5Tinsert(ty, "DMMass_<r (Msun)", offsetof(ClusterData, dark_matter_mass), H5T_NATIVE_DOUBLE);
        H5Tinsert(ty, "radius_hm (kpc)", offsetof(ClusterData, half_mass_rad), H5T_NATIVE_DOUBLE);
        H5Tinsert(ty, "virialized", offsetof(ClusterData, virialized), H5T_NATIVE_DOUBLE);

        return ty;
    }
};

// return: [offset, count]
std::tuple<size_t, size_t> GetRankTaskDivision(size_t total_size, size_t num_proc, size_t rank) {
    // naive division of tasks between processes,
    size_t count = total_size / num_proc;
    // ... there's some remainder
    size_t remainder = total_size % num_proc;

    // if there are 10 processes with a remainder of 4,
    // the first 4 processes should have one extra,
    // and the remaining 6 have the normal amount

    size_t offset;

    // if it's a process that should have an extra one
    if (rank < remainder) {
        offset = rank * (count + 1);
    } else {
        // [ 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 ]
        // < #0 (2+1)  | #1 (2+1)  | #2 (2)| #3 (2)>

        // > rank = 2, remainder 2, count 2;
        // extra = 2 * (2 + 1) = 6
        // normal = (2 - 2) * 2 = 0
        // -> #2 starts at 6 + 0 = 6

        // > rank = 3, remainder 2, count 2;
        // extra = 2 * (2 + 1) = 6
        // normal = (3 - 2) * 2 = 2
        // -> #3 starts at 6 + 2 = 8

        size_t proc_with_extra = remainder * (count + 1);

        size_t proc_with_normal = (rank - remainder) * count;

        offset = proc_with_extra + proc_with_normal;
    }

    size_t this_ct = (rank < remainder) ? count + 1 : count;

    return { offset, this_ct };
}

int main(int argc, char** argv) {
    // 1. initialize MPI env
    MPI_Init(&argc, &argv);

    // credit: https://github.com/astro-claire/arepo-clusters/tree/main
    // ΛCDM star clusters at cosmic dawn: stellar densities, environment, and equilibrium
    // Williams et al (2025)
    std::filesystem::path file_name = "./data/starcluster_data.hdf5";

    int mpi_rank, mpi_size;

    // MPI_COMM_WORLD -> communicator that lets all processes communicate

    // unique id of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 2. property list -> configuration on how HDF5 should work
    // this creates a property list relating to file access
    hid_t fap_list_id = H5Pcreate(H5P_FILE_ACCESS);
    // sets file access property list, uses MPI I/O
    H5Pset_fapl_mpio(fap_list_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // 3. open the file! (each process opens it) with the MPI property list
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, fap_list_id);
    // no more file access needed
    H5Pclose(fap_list_id);

    if (file_id < 0) {
        throw std::runtime_error("Failed to open / create file");
    }

    // mpio transfer prop list is only needed for data transfer,
    // not opening handle of dataset -> H5P_DEFAULT
    hid_t dataset = H5Dopen(file_id, "/redshift_12", H5P_DEFAULT);

    // 4. check size of dataspace & divide tasks
    hid_t dataspace = H5Dget_space(dataset);

    if (H5Sget_simple_extent_ndims(dataspace) != 1) {
        throw std::runtime_error("dataset should be one dimensional");
    }

    hsize_t cluster_ct;
    H5Sget_simple_extent_dims(dataspace, &cluster_ct, nullptr);

    auto [offset, count] = GetRankTaskDivision(cluster_ct, mpi_size, mpi_rank);

    // 5. select hyperslab (contiguous slice of dataset)
    std::vector<ClusterData> cluster_slice(count);

    hsize_t slab_start[1] = { offset };
    hsize_t slab_count[1] = { count };

    H5Sselect_hyperslab(
        dataspace,
        // H5S_SELECT_SET -> replace with new selection
        // (as opposed to logical set ops: and/union/or)
        H5S_SELECT_SET,
        slab_start,
        nullptr,
        slab_count,
        nullptr
    );

    // so hdf5 knows the shape of the buffer
    hid_t slab_space = H5Screate_simple(1, slab_count, slab_count);

    // 6. read with collective I/O
    // plist for dataset transfer
    hid_t transfer_plist = H5Pcreate(H5P_DATASET_XFER);
    // sets data transfew to use collective MPI I/O,
    // all processes participate, MPI handles coordinating
    H5Pset_dxpl_mpio(transfer_plist, H5FD_MPIO_COLLECTIVE);

    hid_t datatype = ClusterData::H5Type();

    H5Dread(
        dataset,
        datatype,
        // write into buf with shape `slab_space`
        slab_space,
        // dataspace of full dataset
        dataspace,
        transfer_plist,
        cluster_slice.data()
    );

    // 7. calculate baryonic fraction for each cluster
    // (how much of the total mass isn't dark matter)
    std::vector<double> baryonic_fraction(count);

    for (size_t i = 0; i < baryonic_fraction.size(); ++i) {
        double baryonic_matter = cluster_slice[i].gas_mass + cluster_slice[i].stellar_mass;
        double total_matter = baryonic_matter + cluster_slice[i].dark_matter_mass;

        baryonic_fraction[i] = total_matter == 0.0 ? 0.0 : baryonic_matter / total_matter;
    }

    // 8. create a group to save the data in
    // the first proc creates it, and the rest open it
    hid_t bfrac_group;
    std::string_view bfrac_group_name = "/baryonic_fraction";

    if (mpi_rank == 0) {
        htri_t group_exists = H5Lexists(file_id, bfrac_group_name.data(), H5P_DEFAULT);

        if (group_exists < 0) {
            throw std::runtime_error("couldn't check if group exists");
        }

        if (group_exists == 0) {
            bfrac_group = H5Gcreate(file_id, bfrac_group_name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            if (bfrac_group < 0) {
                throw std::runtime_error("failed to create group");
            }
        } else {
            bfrac_group = H5Gopen(file_id, bfrac_group_name.data(), H5P_DEFAULT);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        bfrac_group = H5Gopen(file_id, bfrac_group_name.data(), H5P_DEFAULT);
    }

    // 9. create dataset, first proc creates, rest open
    hid_t bfrac_dataset;
    hid_t bfrac_datatype = H5T_NATIVE_DOUBLE;

    if (mpi_rank == 0) {
        bfrac_dataset = H5Dcreate(bfrac_group /* NOLINT */, "/redshift_12", bfrac_datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT,H5P_DEFAULT);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        bfrac_dataset = H5Dopen(bfrac_group /* NOLINT */, "/redshift_12", H5P_DEFAULT);
    }

    // 10. write to dataset through hyperslab
    H5Sselect_hyperslab(
        dataspace,
        H5S_SELECT_SET,
        slab_start,
        nullptr,
        slab_count,
        nullptr
    );

    H5Dwrite(
        bfrac_dataset, // NOLINT
        bfrac_datatype,
        // write into buf with shape `buf_shape`
        slab_space,
        // dataspace of full dataset
        dataspace,
        transfer_plist,
        baryonic_fraction.data()
    );

    // 11. close everything! TODO: RAII wrappers?
    H5Sclose(dataspace);
    H5Sclose(slab_space);

    H5Tclose(datatype);
    H5Pclose(transfer_plist);

    H5Dclose(dataset);
    H5Dclose(bfrac_dataset); // NOLINT

    H5Gclose(bfrac_group); // NOLINT

    H5Fclose(file_id);

    MPI_Finalize();
}
