@echo off
setlocal

set CXX=g++
set CXXFLAGS=-O3 -shared -fPIC -mavx2 -mfma -Wall -Wextra -static -std=c++17
set OUT=vectorforge_core.dll

echo [BUILD] Compiling VectorForge Core...
if not exist build mkdir build

%CXX% %CXXFLAGS% src/core/distance.cpp src/core/hnsw.cpp src/core/ivf_pq.cpp src/core/bridge.cpp src/core/wal.cpp src/core/storage.cpp -o build/%OUT%

if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    exit /b %errorlevel%
)

echo [SUCCESS] Built build\%OUT%
endlocal
