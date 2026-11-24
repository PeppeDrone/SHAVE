/* Tests only: LD_PRELOAD shim that makes a folder behave like a network filesystem that refuses
 * chmod/chown/utime/hardlinks with EPERM. Only paths under $NOCHMOD_PREFIX are affected.
 * Build: cc -shared -fPIC -O1 -o nochmod_shim.so nochmod_shim.c -ldl */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <utime.h>

static int under(const char *p) {
    const char *pre = getenv("NOCHMOD_PREFIX");
    if (!pre || !*pre || !p) return 0;
    return strncmp(p, pre, strlen(pre)) == 0;
}
static int fd_under(int fd) {
    char l[64], t[PATH_MAX];
    snprintf(l, sizeof l, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(l, t, sizeof t - 1);
    if (n <= 0) return 0;
    t[n] = 0;
    return under(t);
}
static int at_under(int dirfd, const char *path) {
    char base[PATH_MAX], full[2 * PATH_MAX];
    if (!path || !*path) return fd_under(dirfd);
    if (path[0] == '/') return under(path);
    if (dirfd == AT_FDCWD) { if (!getcwd(base, sizeof base)) return 0; }
    else {
        char l[64]; snprintf(l, sizeof l, "/proc/self/fd/%d", dirfd);
        ssize_t n = readlink(l, base, sizeof base - 1); if (n <= 0) return 0; base[n] = 0;
    }
    snprintf(full, sizeof full, "%s/%s", base, path);
    return under(full);
}
#define DENY do { errno = EPERM; return -1; } while (0)
#define REAL(name) static __typeof__(&name) real_##name; if (!real_##name) real_##name = (__typeof__(&name))dlsym(RTLD_NEXT, #name)

int chmod(const char *p, mode_t m) { REAL(chmod); if (at_under(AT_FDCWD, p)) DENY; return real_chmod(p, m); }
int fchmod(int fd, mode_t m) { REAL(fchmod); if (fd_under(fd)) DENY; return real_fchmod(fd, m); }
int fchmodat(int d, const char *p, mode_t m, int f) { REAL(fchmodat); if (at_under(d, p)) DENY; return real_fchmodat(d, p, m, f); }
int chown(const char *p, uid_t u, gid_t g) { REAL(chown); if (at_under(AT_FDCWD, p)) DENY; return real_chown(p, u, g); }
int lchown(const char *p, uid_t u, gid_t g) { REAL(lchown); if (at_under(AT_FDCWD, p)) DENY; return real_lchown(p, u, g); }
int fchown(int fd, uid_t u, gid_t g) { REAL(fchown); if (fd_under(fd)) DENY; return real_fchown(fd, u, g); }
int fchownat(int d, const char *p, uid_t u, gid_t g, int f) { REAL(fchownat); if (at_under(d, p)) DENY; return real_fchownat(d, p, u, g, f); }
int utime(const char *p, const struct utimbuf *t) { REAL(utime); if (at_under(AT_FDCWD, p)) DENY; return real_utime(p, t); }
int utimes(const char *p, const struct timeval t[2]) { REAL(utimes); if (at_under(AT_FDCWD, p)) DENY; return real_utimes(p, t); }
int lutimes(const char *p, const struct timeval t[2]) { REAL(lutimes); if (at_under(AT_FDCWD, p)) DENY; return real_lutimes(p, t); }
int futimes(int fd, const struct timeval t[2]) { REAL(futimes); if (fd_under(fd)) DENY; return real_futimes(fd, t); }
int futimens(int fd, const struct timespec t[2]) { REAL(futimens); if (fd_under(fd)) DENY; return real_futimens(fd, t); }
int utimensat(int d, const char *p, const struct timespec t[2], int f) { REAL(utimensat); if (at_under(d, p)) DENY; return real_utimensat(d, p, t, f); }
int link(const char *a, const char *b) { REAL(link); if (at_under(AT_FDCWD, b)) DENY; return real_link(a, b); }
int linkat(int d1, const char *a, int d2, const char *b, int f) { REAL(linkat); if (at_under(d2, b)) DENY; return real_linkat(d1, a, d2, b, f); }
