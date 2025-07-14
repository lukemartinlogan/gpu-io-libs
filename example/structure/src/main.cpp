#include <cstdio>
#include <cstdint>

using offset_t = uint64_t;
using len_t = uint64_t;

// Superblock v2-3
struct Superblock {
    // 0x 89 484446 0d0a1a0a
    //       "HDF"
    uint8_t signature[8];
    // 0x02
    uint8_t version_number;
    // number of bytes used for storing addresses,
    // addresses are relative to base address,
    // superblock address is usually base
    // !! this struct assumes this is 0x08 !!
    uint8_t size_of_offsets;
    // num of bytes used to store size of object
    // !! this struct assumes this is 0x08 !!
    uint8_t size_of_lengths;
    // b0: file opened for write access
    // b1: reserved
    // b2: file opened for SWMR
    // b3-7: reserved
    uint8_t file_consistency_flags;
    // file address of first byte of HDF5 data
    // generally 0x00, nonzero for 'driver cases'?
    offset_t base_addr;
    // address of extra superblock metadata
    // if undefined -> 0xff...fff
    offset_t superblock_ext;
    // address of first bytes past all the HDF5 data,
    // can be used to check for accidental truncation
    offset_t eof_addr;
    // address of data objects, entry point into group graph
    offset_t root_group_header_addr;
    // Jenkins' lookup3 -> https://www.burtleburtle.net/bob/hash/doobs.html
    uint32_t checksum;
};

// TODO: defined in Superblock::superblock_ext
struct SharedObjectHeaderMessageTable {};

// TODO: defined in Superblock::superblock_ext
struct BTreeKValuesMessage {};

// TODO: defined in Superblock::superblock_ext
struct DriverInfoMessage {};

// TODO: defined in Superblock::superblock_ext
struct FileSpaceInfoMessage {};

struct BTreeNode {
    // 0x54524545 -> b"TREE"
    uint8_t signature[4];
    // 0x00 -> points to group nodes
    // 0x01 -> points to raw data chunk
    // this can be enum
    uint8_t ty;
    // what tree level this node is at
    // root -> 0
    uint8_t level;
    // number of children to which this node points
    uint16_t entries_used;
    // relative addr of left sibling, if leftmost
    // then undefined addr -> 0xff...fff
    offset_t left_addr;
    // relative addr of right sibling, if rightmost
    // then undefined addr -> 0xff...fff
    offset_t right_addr;

    // ...
};



int main() {
}