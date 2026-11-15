#include "Rachis/rachis.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check(rachis_status status, const rachis_error* error, const char* operation)
{
    if (status == RACHIS_STATUS_OK)
    {
        return 1;
    }
    fprintf(stderr, "%s failed: %u (%s)\n", operation, status, error->message);
    return 0;
}

int main(void)
{
    const uint8_t expected[] = {0x13, 0x57, 0x9b, 0xdf, 0x24, 0x68, 0xac, 0xe0};
    uint8_t actual[sizeof(expected)] = {0};
    rachis_error error = {.struct_size = sizeof(rachis_error)};
    rachis_device* device = NULL;
    rachis_handle upload = 0;
    rachis_handle readback = 0;
    rachis_handle pool = 0;
    rachis_handle command = 0;
    rachis_handle timeline = 0;
    rachis_device_desc deviceDesc = {
        .struct_size = sizeof(rachis_device_desc),
        .abi_version = RACHIS_ABI_VERSION,
        .backend = RACHIS_SMOKE_BACKEND,
        .adapter_preference = RACHIS_SMOKE_ADAPTER,
        .validation_level = RACHIS_VALIDATION_DISABLED,
        .application_name = "Rachis C ABI smoke",
        .engine_name = "Rachis",
    };
    rachis_buffer_desc bufferDesc = {
        .struct_size = sizeof(rachis_buffer_desc),
        .abi_version = RACHIS_ABI_VERSION,
        .size = sizeof(expected),
        .usage = RACHIS_BUFFER_COPY_SOURCE,
        .memory_type = RACHIS_MEMORY_UPLOAD,
    };
    rachis_command_pool_desc poolDesc = {
        .struct_size = sizeof(rachis_command_pool_desc),
        .abi_version = RACHIS_ABI_VERSION,
        .queue_type = RACHIS_QUEUE_COPY,
        .max_command_buffers = 1,
    };
    int ok = 0;

    if (rachis_abi_version() != RACHIS_ABI_VERSION)
    {
        goto cleanup;
    }
    {
        const rachis_status created = rachis_device_create(&deviceDesc, &device, &error);

        if (created == RACHIS_STATUS_BACKEND_UNAVAILABLE)
        {
            fprintf(stderr, "create device skipped: %u (%s)\n", created, error.message);
            rachis_device_destroy(device);
            return 0;
        }
        if (!check(created, &error, "create device"))
        {
            goto cleanup;
        }
    }
    if (!check(rachis_buffer_create(device, &bufferDesc, &upload, &error), &error, "create upload"))
    {
        goto cleanup;
    }
    bufferDesc.usage = RACHIS_BUFFER_COPY_DESTINATION;
    bufferDesc.memory_type = RACHIS_MEMORY_READBACK;
    if (!check(rachis_buffer_create(device, &bufferDesc, &readback, &error), &error, "create readback") ||
        !check(rachis_buffer_write(device, upload, 0, expected, sizeof(expected), &error), &error, "write upload") ||
        !check(rachis_command_pool_create(device, &poolDesc, &pool, &error), &error, "create pool") ||
        !check(rachis_command_acquire(device, pool, &command, &error), &error, "acquire command") ||
        !check(rachis_command_copy_buffer(device, command, upload, readback, 0, 0, sizeof(expected), &error),
               &error,
               "record copy") ||
        !check(rachis_command_submit(device, command, &timeline, &error), &error, "submit") ||
        !check(rachis_timeline_wait(device, timeline, &error), &error, "wait") ||
        !check(rachis_buffer_read(device, readback, 0, actual, sizeof(actual), &error), &error, "readback"))
    {
        goto cleanup;
    }
    if (memcmp(expected, actual, sizeof(expected)) != 0)
    {
        fprintf(stderr, "C ABI copy contents differ\n");
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (timeline != 0)
    {
        (void)rachis_timeline_release(device, timeline, &error);
    }
    if (command != 0 && timeline == 0)
    {
        (void)rachis_command_discard(device, command, &error);
    }
    if (pool != 0)
    {
        (void)rachis_command_pool_destroy(device, pool, &error);
    }
    if (readback != 0)
    {
        (void)rachis_buffer_destroy(device, readback, &error);
    }
    if (upload != 0)
    {
        (void)rachis_buffer_destroy(device, upload, &error);
    }
    rachis_device_destroy(device);
    return ok ? 0 : 1;
}
