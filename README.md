# xgboost_sharp
Unofficial C# wrapper for the awesome XGBoost library (https://github.com/dmlc/xgboost).

The makefiles assume xgboost is installed on your system.  As there isn't a 'make install' for that, you can do the following to ensure it is available on Linux (for Windows, I recommend using VC++ 2015 Community edition and letting the IDE handle your paths).

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
