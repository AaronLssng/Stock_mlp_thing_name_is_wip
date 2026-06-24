// headers
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#include <io.h>
#include <sys/stat.h>
#define mkdir(path, mode) _mkdir(path)
#define stat _stat
#define strdup _strdup
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>

// Windows DLL export
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// Error codes
#define STOCK_SUCCESS 0
#define STOCK_ERR_FILE_NOT_FOUND -1
#define STOCK_ERR_IO -2
#define STOCK_ERR_INVALID_HEADER -3
#define STOCK_ERR_OUT_OF_RANGE -4
#define STOCK_ERR_MEMORY -5
#define STOCK_ERR_INVALID_ARG -6
#define STOCK_ERR_FILE_EXISTS -7

// structs
typedef struct
{
    char stock_name[16];     // Stock symbol, null-terminated
    char timeframe[8];       // "1d", "1h", "15m", etc.
    uint64_t num_bars;       // Number of bars in file
    int64_t start_timestamp; // Unix timestamp of first bar
    int64_t end_timestamp;   // Unix timestamp of last bar
    char reserved[16];       // Padding for future use
} StockHeader;

typedef struct
{
    int64_t timestamp; // Unix timestamp
    float open;
    float high;
    float low;
    float close;
    double volume; // Using double (8 bytes) for volume
} StockBar;

// Safe function to set stock name
void set_stock_name(StockHeader *header, const char *name)
{
    strncpy(header->stock_name, name, sizeof(header->stock_name) - 1);
    header->stock_name[sizeof(header->stock_name) - 1] = '\0';
}

// Safe function to set timeframe
void set_timeframe(StockHeader *header, const char *tf)
{
    strncpy(header->timeframe, tf, sizeof(header->timeframe) - 1);
    header->timeframe[sizeof(header->timeframe) - 1] = '\0';
}

// Helper function to create directories recursively
static int create_directories(const char *filepath)
{
    char *path = strdup(filepath);
    if (path == NULL)
        return STOCK_ERR_MEMORY;

    // Start from the beginning to handle drive letters on Windows (e.g., C:\)
    char *p = path;

// Skip drive letter on Windows (e.g., "C:\")
#ifdef _WIN32
    if (strlen(path) >= 2 && path[1] == ':')
        p = path + 2;
#endif

    for (; *p; p++)
    {
        if (*p == '/' || *p == '\\')
        {
            *p = '\0';

            // Skip empty strings and drive roots
            if (strlen(path) > 0 &&
#ifdef _WIN32
                !(strlen(path) == 2 && path[1] == ':')
#else
                strlen(path) > 1
#endif
            )
            {
#ifdef _WIN32
                struct _stat st;
                if (_stat(path, &st) != 0)
                {
                    if (_mkdir(path) != 0 && errno != EEXIST)
                    {
                        free(path);
                        return STOCK_ERR_IO;
                    }
                }
#else
                struct stat st;
                if (stat(path, &st) != 0)
                {
                    if (mkdir(path, 0755) != 0)
                    {
                        free(path);
                        return STOCK_ERR_IO;
                    }
                }
#endif
            }

// Restore the original character
#ifdef _WIN32
            *p = '\\'; // Use backslash on Windows
#else
            *p = '/'; // Use forward slash on Unix
#endif
        }
    }
    free(path);
    return STOCK_SUCCESS;
}

DLL_EXPORT int stock_create_file(const char *filepath, const char *stock_name, const char *timeframe)
{
    // Validate inputs
    if (filepath == NULL || stock_name == NULL || timeframe == NULL)
    {
        return STOCK_ERR_INVALID_ARG;
    }

    // Validate stock_name length (max 15 chars + null terminator)
    if (strlen(stock_name) > 15)
    {
        return STOCK_ERR_INVALID_ARG;
    }

    // Validate timeframe length (max 7 chars + null terminator)
    if (strlen(timeframe) > 7)
    {
        return STOCK_ERR_INVALID_ARG;
    }

    // Create directories recursively
    int result = create_directories(filepath);
    if (result != STOCK_SUCCESS)
    {
        return result;
    }

// Open file for writing (create new, fail if exists)
#ifdef _WIN32
    int fd = _open(filepath, _O_CREAT | _O_WRONLY | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    int fd = open(filepath, O_CREAT | O_WRONLY | O_EXCL, 0644);
#endif

    if (fd == -1)
    {
        if (errno == EEXIST)
        {
            return STOCK_ERR_FILE_EXISTS;
        }
        return STOCK_ERR_IO;
    }

    // Initialize header
    StockHeader header1;
    memset(&header1, 0, sizeof(StockHeader));

    set_stock_name(&header1, stock_name);
    set_timeframe(&header1, timeframe);
    header1.num_bars = 0;
    header1.start_timestamp = 0;
    header1.end_timestamp = 0;

#ifdef _WIN32
    ssize_t bytes_written = _write(fd, &header1, sizeof(StockHeader));
#else
    ssize_t bytes_written = write(fd, &header1, sizeof(StockHeader));
#endif

    if (bytes_written != sizeof(StockHeader))
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    return STOCK_SUCCESS;
}

DLL_EXPORT int stock_append_bars(const char *filepath, const StockBar *bars, size_t num_bars)
{
    if (filepath == NULL || bars == NULL || num_bars == 0)
    {
        return STOCK_ERR_INVALID_ARG;
    }

#ifdef _WIN32
    int fd = _open(filepath, _O_RDWR | _O_BINARY);
#else
    int fd = open(filepath, O_RDWR);
#endif

    if (fd == -1)
    {
        if (errno == ENOENT)
        {
            return STOCK_ERR_FILE_NOT_FOUND;
        }
        return STOCK_ERR_IO;
    }

    // Read header
    StockHeader header;
#ifdef _WIN32
    ssize_t bytes_read = _read(fd, &header, sizeof(StockHeader));
#else
    ssize_t bytes_read = read(fd, &header, sizeof(StockHeader));
#endif

    if (bytes_read != sizeof(StockHeader))
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_INVALID_HEADER;
    }

#ifdef _WIN32
    off_t file_size = _lseek(fd, 0, SEEK_END);
#else
    off_t file_size = lseek(fd, 0, SEEK_END);
#endif

    if (file_size == -1)
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

    size_t bytes_to_write = num_bars * sizeof(StockBar);
    ssize_t written;
#ifdef _WIN32
    written = _write(fd, bars, bytes_to_write);
#else
    written = write(fd, bars, bytes_to_write);
#endif

    if (written != (ssize_t)bytes_to_write)
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

    header.num_bars += num_bars;
    if (header.num_bars == num_bars)
    {
        header.start_timestamp = bars[0].timestamp;
    }
    header.end_timestamp = bars[num_bars - 1].timestamp;

#ifdef _WIN32
    if (_lseek(fd, 0, SEEK_SET) == -1)
#else
    if (lseek(fd, 0, SEEK_SET) == -1)
#endif
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

    ssize_t bytes_written;
#ifdef _WIN32
    bytes_written = _write(fd, &header, sizeof(StockHeader));
#else
    bytes_written = write(fd, &header, sizeof(StockHeader));
#endif

    if (bytes_written != sizeof(StockHeader))
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    return STOCK_SUCCESS;
}

DLL_EXPORT int stock_read_all(const char *filepath, StockBar **out_bars, size_t *out_count)
{
    if (filepath == NULL || out_bars == NULL || out_count == NULL)
    {
        return STOCK_ERR_INVALID_ARG;
    }

#ifdef _WIN32
    int fd = _open(filepath, _O_RDONLY | _O_BINARY);
#else
    int fd = open(filepath, O_RDONLY);
#endif

    if (fd == -1)
    {
        if (errno == ENOENT)
        {
            return STOCK_ERR_FILE_NOT_FOUND;
        }
        return STOCK_ERR_IO;
    }

    StockHeader header;
#ifdef _WIN32
    ssize_t read_bytes = _read(fd, &header, sizeof(StockHeader));
#else
    ssize_t read_bytes = read(fd, &header, sizeof(StockHeader));
#endif

    if (read_bytes != sizeof(StockHeader))
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_INVALID_HEADER;
    }

    size_t num_bars = header.num_bars;

    if (num_bars == 0)
    {
        *out_bars = NULL;
        *out_count = 0;
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_SUCCESS;
    }

    *out_bars = (StockBar *)malloc(num_bars * sizeof(StockBar));
    if (*out_bars == NULL)
    {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_MEMORY;
    }

    size_t total_bytes = num_bars * sizeof(StockBar);

    ssize_t bytes_read;
#ifdef _WIN32
    bytes_read = _read(fd, *out_bars, total_bytes);
#else
    bytes_read = read(fd, *out_bars, total_bytes);
#endif

    if (bytes_read != (ssize_t)total_bytes)
    {

        if (bytes_read == -1)
        {
            perror("read failed");
        }
        else
        {
            fprintf(stderr, "Warning: only %zd of %zu Bytes read.\n", bytes_read, total_bytes);
        }
        free(*out_bars);
        *out_bars = NULL;
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return STOCK_ERR_IO;
    }

    *out_count = num_bars;
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif

    return STOCK_SUCCESS;
}

DLL_EXPORT void stock_free_bars(StockBar *bars)
{
    if (bars != NULL)
    {
        free(bars);
    }
}