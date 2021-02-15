
(:goal 
    (and 
        (scrubbed ?floor1) 
        (scrubbed ?floor2) 
        (and 
            (soaked ?swab1) 
            (inside ?swab1 ?cabinet1)
        ) 
        (inside ?bucket1 ?cabinet1) 
        (inside ?soap1 ?cabinet1)
    )
)