if [[ -d build ]] ; then
    rm -r build
    echo "Removed build directory"
fi

mkdir build && cd build && cmake ../ && make -j4
