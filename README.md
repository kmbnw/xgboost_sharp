# xgboost_sharp
Unofficial C# wrapper for the awesome XGBoost library (https://github.com/dmlc/xgboost).

The C# class calls a C++ class (to keep the interop interface simple); you may find the C++ to be directly useful as well.

The makefiles in this project assume xgboost is installed on your system.  As there isn't a 'make install' for that, you can do the following to ensure it is available on Linux (for Windows, I recommend using VC++ 2015 Community edition and letting the IDE handle your paths).

1. Clone xgboost from https://github.com/dmlc/xgboost
2. Compile xgboost following the instructions at https://xgboost.readthedocs.io/en/latest/build.html
3. Copy the lib and include files out to /usr/local.  From the directory you cloned xgboost to, run the following:

```bash
sudo cp lib/libxgboost.so /usr/local/lib/

# not actually sure if this is needed but xgboost itself references rabit/c_api.h so...
sudo cp lib/rabit/librabit.a /usr/local/lib/

sudo cp -R include/xgboost /usr/local/include/
sudo cp -R rabit/include/rabit /usr/local/include/
sudo cp -R rabit/include/dmlc /usr/local/include/
```

Assuming now that /usr/local is in your library and include paths (which seems like a safe assumption) you should be able to compile and run **this** project.

== Running on Windows ==
This *does* work on Windows, compiled with VC++ (tested on Windows 7, Windows 10, and Windows Server 2012).  I don't happen to have a Windows machine anymore, but I believe the following should get it working:
 * Change the .so to .dll in the C# wrapper
 * You may also need to remove the [return: MarshalAs(UnmanagedType.I1)] lines from the C# wrapper
 * Add **__declspec** to the C API functions
I'm happy to accept pull requests for this.
