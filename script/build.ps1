$buildMode = $Args[0]
if (-Not($buildMode -eq "Debug" -or $buildMode -eq "Release")) {
    echo "current build mode is $buildMode"
    echo "please use 'build.ps1 Debug' or 'build.ps1 Release'"
    return
}

$generator="Visual Studio 16 2019"
$binaryFolder = ".\build\$buildMode\"

if ($buildMode -eq "Release") {
    rm $binaryFolder -Force -Recurse
}
mkdir $binaryFolder\

cmake `
    -A x64 `
    -G $generator `
    -DCMAKE_BUILD_TYPE="$buildMode" `
    -B "$binaryFolder" `
    -S .\
cmake --build "$binaryFolder" --config "$buildMode"