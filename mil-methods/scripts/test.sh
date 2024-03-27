

echo "参数1： $1"

if [ "$1" == "gc" ]; then
    echo "执行gc"
elif [ "$1" == "ngc" ]; then
    echo "执行ngc"
fi

