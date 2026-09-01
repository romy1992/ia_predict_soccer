Param(
    [string]$ApiBase = "http://127.0.0.1:8000"
)

Write-Host "[1/3] GET /health"
Invoke-RestMethod -Method GET -Uri "$ApiBase/health" | ConvertTo-Json -Depth 5

Write-Host "[2/3] GET /markets"
Invoke-RestMethod -Method GET -Uri "$ApiBase/markets" | ConvertTo-Json -Depth 5

Write-Host "[3/3] GET /jobs/history?limit=5"
Invoke-RestMethod -Method GET -Uri "$ApiBase/jobs/history?limit=5" | ConvertTo-Json -Depth 5

