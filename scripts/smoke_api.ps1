Param(
    [string]$ApiBase = "http://127.0.0.1:8000"
)

Write-Host "[1/6] GET /health"
Invoke-RestMethod -Method GET -Uri "$ApiBase/health" | ConvertTo-Json -Depth 5

Write-Host "[2/6] GET /markets"
Invoke-RestMethod -Method GET -Uri "$ApiBase/markets" | ConvertTo-Json -Depth 5

Write-Host "[3/6] GET /jobs/history?limit=5"
Invoke-RestMethod -Method GET -Uri "$ApiBase/jobs/history?limit=5" | ConvertTo-Json -Depth 5

Write-Host "[4/6] GET /dashboard/overview"
Invoke-RestMethod -Method GET -Uri "$ApiBase/dashboard/overview" | ConvertTo-Json -Depth 5

Write-Host "[5/6] GET /dashboard/day?limit=3"
Invoke-RestMethod -Method GET -Uri "$ApiBase/dashboard/day?limit=3" | ConvertTo-Json -Depth 6

Write-Host "[6/6] GET /dashboard/match/0"
Invoke-RestMethod -Method GET -Uri "$ApiBase/dashboard/match/0" | ConvertTo-Json -Depth 6





