#pragma once

#if defined _WIN32 || defined __CYGWIN__
#  define G1RLQPController_DLLIMPORT __declspec(dllimport)
#  define G1RLQPController_DLLEXPORT __declspec(dllexport)
#  define G1RLQPController_DLLLOCAL
#else
// On Linux, for GCC >= 4, tag symbols using GCC extension.
#  if __GNUC__ >= 4
#    define G1RLQPController_DLLIMPORT __attribute__((visibility("default")))
#    define G1RLQPController_DLLEXPORT __attribute__((visibility("default")))
#    define G1RLQPController_DLLLOCAL __attribute__((visibility("hidden")))
#  else
// Otherwise (GCC < 4 or another compiler is used), export everything.
#    define G1RLQPController_DLLIMPORT
#    define G1RLQPController_DLLEXPORT
#    define G1RLQPController_DLLLOCAL
#  endif // __GNUC__ >= 4
#endif // defined _WIN32 || defined __CYGWIN__

#ifdef G1RLQPController_STATIC
// If one is using the library statically, get rid of
// extra information.
#  define G1RLQPController_DLLAPI
#  define G1RLQPController_LOCAL
#else
// Depending on whether one is building or using the
// library define DLLAPI to import or export.
#  ifdef G1RLQPController_EXPORTS
#    define G1RLQPController_DLLAPI G1RLQPController_DLLEXPORT
#  else
#    define G1RLQPController_DLLAPI G1RLQPController_DLLIMPORT
#  endif // G1RLQPController_EXPORTS
#  define G1RLQPController_LOCAL G1RLQPController_DLLLOCAL
#endif // G1RLQPController_STATIC
