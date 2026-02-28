#pragma once

#include "container.h"
#include "ref.h"
#include "context.h"
#include "error.h"

namespace kvhdf5 {

// Forward declaration - Group<B> will be defined in hdf5_group.h
template<RawBlobStore B>
class Group;

/**
 * File<B> is the entry point to the kvhdf5 API.
 *
 * It uniquely owns a Container<B>. All other handles (Group, Dataset,
 * Attribute) are derived from it and hold non-owning Ref<Container<B>>
 * references. The File must outlive all handles derived from it.
 */
template<RawBlobStore B>
class File {
    Container<B> container_;

    explicit File(Container<B>&& container)
        : container_(std::move(container)) {}

public:
    // Not copyable (owns the container)
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    // Movable
    File(File&&) = default;
    File& operator=(File&&) = default;

    /**
     * Create a new File with a fresh Container.
     * @param blob_store  The blob store backend (moved into the container).
     * @param ctx         Context providing an allocator.
     * @return            The new File on success, or an Error.
     */
    static expected<File> Create(B&& blob_store, Context ctx) {
        Container<B> c(std::move(blob_store), ctx.allocator_);
        return File(std::move(c));
    }

    /**
     * Open the root group.
     * Declared here; defined in hdf5_group.h after Group<B> is complete.
     */
    // Group<B> OpenRootGroup();

    /**
     * Get mutable access to the underlying Container.
     */
    CROSS_FUN Container<B>& GetContainer() { return container_; }

    /**
     * Get const access to the underlying Container.
     */
    CROSS_FUN const Container<B>& GetContainer() const { return container_; }
};

} // namespace kvhdf5
