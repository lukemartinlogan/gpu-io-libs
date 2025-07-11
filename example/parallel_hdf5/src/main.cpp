#include <hdf5.h>
#include <mpi.h>
#include <string>
#include <filesystem>

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

int main(int argc, char** argv) {
    // 1. initialize MPI env
    MPI_Init(&argc, &argv);

    std::filesystem::path file_name = "./data/starcluster_data.hdf5";

    int rank, size;

    // MPI_COMM_WORLD -> communicator that lets all processes communicate

    // unique id of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // 4. check size of dataspace
    hid_t dataspace = H5Dget_space(dataset);

    if (H5Sget_simple_extent_ndims(dataspace) != 1) {
        throw std::runtime_error("dataset should be one dimensional");
    }

    hsize_t cluster_ct;
    H5Sget_simple_extent_dims(dataspace, &cluster_ct, nullptr);
    H5Sclose(dataspace);
}
