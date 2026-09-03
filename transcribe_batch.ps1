param(
    [ValidateSet("tiny", "base", "small", "medium", "large")]
    [string]$Model = "base",
    
    [ValidateSet("txt", "srt", "vtt", "json")]
    [string]$Format = "txt"
)

$projectFolder = $PSScriptRoot
$videosFolder = Join-Path $projectFolder "AEX_videos"
$venvPython = Join-Path $projectFolder "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCommand = $venvPython
} else {
    $pythonCommand = "python"
}
$env:PYTHONIOENCODING = "utf-8"
$videos = Get-ChildItem -Path $videosFolder -Filter "*.mp4" | Sort-Object Name
$totalVideos = $videos.Count
$successCount = 0
$failCount = 0
Write-Host "==== BATCH VIDEO TRANSCRIPTION ====" -ForegroundColor Cyan
Write-Host "Folder: $videosFolder" -ForegroundColor Yellow
Write-Host "Total videos: $totalVideos" -ForegroundColor Yellow
Write-Host "Model: $Model" -ForegroundColor Yellow
Write-Host "Format: $Format" -ForegroundColor Yellow
Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Yellow
Write-Host ""
$startTime = Get-Date
foreach ($video in $videos) {
    $index = $videos.IndexOf($video) + 1
    $videoPath = Join-Path $videosFolder $video.Name
    $percentage = [math]::Round(($index / $totalVideos) * 100, 0)
    Write-Host "[$index/$totalVideos] ($percentage percent) Processing: $($video.Name)" -ForegroundColor Green
    try {
        $output = & $pythonCommand (Join-Path $projectFolder "transcriptor.py") $videoPath --model $Model --format $Format 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host "  FAILED" -ForegroundColor Red
            $failCount++
        }
    } catch {
        Write-Host "  ERROR" -ForegroundColor Red
        $failCount++
    }
}
$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host ""
Write-Host "==== COMPLETED ====" -ForegroundColor Cyan
Write-Host "Success: $successCount/$totalVideos" -ForegroundColor Green
Write-Host "Failed: $failCount/$totalVideos" -ForegroundColor Red
Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor Yellow
Write-Host ""
