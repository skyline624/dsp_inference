# Run the cocotb FPGA-cluster simulation inside the hdlc/sim:osvb container.
#
#   ./run.ps1            # 1 node  (milestone 1 smoke)
#   ./run.ps1 -N 2       # 2 nodes daisy-chained over UART
#
# Requires Docker Desktop running and the image pulled:
#   docker pull hdlc/sim:osvb
param(
    [int]$N = 1,
    [string]$Module = "test_cluster"
)
$ErrorActionPreference = "Stop"
$repo = ((Resolve-Path "$PSScriptRoot\..").Path) -replace '\\', '/'
Write-Host "Cluster sim -> N=$N  module=$Module  repo=$repo"

# LUT .hex files are loaded by $readmemh with relative paths; the container copies
# them next to the testbench before running, so they resolve from the sim cwd.
docker run --rm -v "${repo}:/work" -w /work/sim hdlc/sim:osvb `
    bash -lc "cp -f ../src/*.hex . && make clean >/dev/null 2>&1; make N=$N MODULE=$Module"
