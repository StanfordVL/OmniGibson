while ($true) {
    dvc repro export_objs@scenes/house_single_floor ;
    dvc repro --glob export_objs@objects* ;
    taskkill /IM 3dsmax.exe /F ;
}