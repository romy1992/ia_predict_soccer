Param(
    [string]$ApiBase = "http://127.0.0.1:8000"
)

Write-Host "[1/5] GET /health"
Invoke-RestMethod -Method GET -Uri "$ApiBase/health" | ConvertTo-Json -Depth 5

Write-Host "[2/5] GET /markets"
Invoke-RestMethod -Method GET -Uri "$ApiBase/markets" | ConvertTo-Json -Depth 5

Write-Host "[3/5] GET /jobs/history?limit=5"
Invoke-RestMethod -Method GET -Uri "$ApiBase/jobs/history?limit=5" | ConvertTo-Json -Depth 5

Write-Host "[4/5] GET /dashboard/overview"
Invoke-RestMethod -Method GET -Uri "$ApiBase/dashboard/overview" | ConvertTo-Json -Depth 5

Write-Host "[5/5] GET /dashboard/day?limit=3"
Invoke-RestMethod -Method GET -Uri "$ApiBase/dashboard/day?limit=3" | ConvertTo-Json -Depth 6




