#ifndef KODES_TYPE_TABLE
#define KODES_TYPE_TABLE

#pragma once

#include <string.h>

#include "basic_types.cuh"

// Choosing a device object by name, at run time.
//
// A `TypeEntry` is one concrete class reduced to plain function pointers: the
// four host statics DeviceObject.cuh gave it, plus what it costs in device
// memory. A table of them is a list of the classes a name may select, and it is
// the whole of the runtime dispatch - no host vtable is involved, so a caller
// compiled by anything (nvc++, plain g++) can select and own a class whose
// virtuals only exist on the device.
//
// The tables themselves live in a .cu of the library, one per abstract base,
// and adding a class to one is a single line.

namespace kodes
{

template<class Base>
struct TypeEntry
{
    const char* name;

    Base* (*create)
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize,
        Base* hostStub
    );

    Base* (*createStub)
    (
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    );

    void (*destroy)(Base* devObject, Base* hostStub);

    void (*destroyStub)(Base* hostStub);

    // What one of these costs, before there is one to ask: device memory per
    // resident thread and per system of the batch. planLaunch() adds up the
    // entries of every table the run will draw from.
    size_t (*scratchBytesPerThread)(const label systemSize, const label parameterSize);

    size_t (*stateBytesPerSystem)(const label systemSize, const label parameterSize);
};

// One line of a table. The lambdas are captureless, so each converts to a plain
// function pointer; all they do is put the static back on its own class and
// cast the base pointer back to what the entry knows it is.
template<class Base, class Derived>
__host__ inline TypeEntry<Base> typeEntry(const char* name)
{
    return TypeEntry<Base>
    {
        name,

        [] (const label b, const label s, const label n, const label p, Base* stub) -> Base*
        {
            return Derived::create(b, s, n, p, static_cast<Derived*>(stub));
        },

        [] (const label b, const label s, const label n, const label p) -> Base*
        {
            return Derived::createStub(b, s, n, p);
        },

        [] (Base* devObject, Base* hostStub)
        {
            Derived::destroy
            (
                static_cast<Derived*>(devObject), static_cast<Derived*>(hostStub)
            );
        },

        [] (Base* hostStub)
        {
            Derived::destroyStub(static_cast<Derived*>(hostStub));
        },

        &Derived::scratchBytesPerThread,
        &Derived::stateBytesPerSystem
    };
}

// The entry `name` selects, or a message listing the ones it could have been.
template<class Base>
__host__ inline const TypeEntry<Base>* findType
(
    const TypeEntry<Base>* table,
    const label size,
    const char* name,
    const char* what
)
{
    for (label i = 0; i < size; ++i)
    {
        if (strcmp(table[i].name, name) == 0)
        {
            return table + i;
        }
    }

    fprintf(stderr, "kodes error at %s:%d: unknown %s \"%s\", known are", __FILE__, __LINE__, what, name);
    for (label i = 0; i < size; ++i)
    {
        fprintf(stderr, " \"%s\"", table[i].name);
    }
    fprintf(stderr, "\n");
    std::exit(EXIT_FAILURE);

    return nullptr;
}

// Owns one device object and the host stub that holds its buffers, and hands
// both back to the class that made them. Which class that was is the entry it
// was built from, so nothing outside has to remember.
template<class Base>
class Handle
{
    const TypeEntry<Base>* type_;

    Base* device_;
    Base* host_;

public:

    __host__ Handle()
    : type_(nullptr), device_(nullptr), host_(nullptr) {}

    __host__ Handle
    (
        const TypeEntry<Base>* type,
        const label batchSize,
        const label scratchSize,
        const label systemSize,
        const label parameterSize
    )
    : type_(type), device_(nullptr), host_(nullptr)
    {
        if (!type_)
        {
            return;
        }

        host_ = type_->createStub(batchSize, scratchSize, systemSize, parameterSize);
        device_ = type_->create(batchSize, scratchSize, systemSize, parameterSize, host_);
    }

    __host__ ~Handle() { clear(); }

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    __host__ Handle(Handle&& other) noexcept
    : type_(other.type_), device_(other.device_), host_(other.host_)
    {
        other.type_ = nullptr;
        other.device_ = nullptr;
        other.host_ = nullptr;
    }

    __host__ Handle& operator=(Handle&& other) noexcept
    {
        if (this != &other)
        {
            clear();

            type_ = other.type_;
            device_ = other.device_;
            host_ = other.host_;

            other.type_ = nullptr;
            other.device_ = nullptr;
            other.host_ = nullptr;
        }

        return *this;
    }

    __host__ void clear()
    {
        if (type_ && device_)
        {
            type_->destroy(device_, host_);
        }

        if (type_ && host_)
        {
            type_->destroyStub(host_);
        }

        type_ = nullptr;
        device_ = nullptr;
        host_ = nullptr;
    }

    // The object the kernels dispatch on
    __host__ Base* device() const { return device_; }

    // Its host side twin, which is what the host side calls go through
    __host__ Base* host() const { return host_; }

    __host__ const char* name() const { return type_ ? type_->name : "none"; }

    __host__ explicit operator bool() const { return device_ != nullptr; }
};

}

#endif
