(define (problem cleaning_cupboards_0)
    (:domain igibson)

    (:objects
     	cabinet1 cabinet2 cabinet3 cabinet4 - cabinet
    	towel1 towel2 - towel
    	atomizer1 atomizer2 - atomizer
    	counter1 - counter
    	soap1 soap2 - soap
    	sink1 - sink
    	scrub_brush1 scrub_brush2 - scrub_brush
    	bucket1 bucket2 - bucket
    	rag1 rag2 rag3 rag4 - rag
    	porcelain1 porcelain2 - porcelain
    	photograph1 photograph2 - photograph
    	console_table1 - console_table
    	towel_rack1 - towel_rack
    )
    
    (:init 
        (or 
            (not 
                (scrubbed cabinet1)
            ) 
            (dusty cabinet1)
        ) 
        (or 
            (not 
                (scrubbed cabinet2)
            ) 
            (dusty cabinet2)
        ) 
        (inside towel1 cabinet1) 
        (inside towel2 cabinet1) 
        (and 
            (not 
                (scrubbed towel1)
            ) 
            (not 
                (scrubbed towel2)
            )
        ) 
        (ontop atomizer1 counter1) 
        (nextto soap1 sink1) 
        (ontop scrub_brush1 counter1) 
        (under bucket1 sink1) 
        (ontop rag1 sink1) 
        (ontop rag2 sink1) 
        (and 
            (not 
                (soaked rag1)
            ) 
            (not 
                (soaked rag2)
            )
        ) 
        (or 
            (not 
                (scrubbed cabinet3)
            ) 
            (dusty cabinet3)
        ) 
        (or 
            (not 
                (scrubbed cabinet4)
            ) 
            (dusty cabinet4)
        ) 
        (inside porcelain1 cabinet3) 
        (inside porcelain2 cabinet3) 
        (and 
            (not 
                (scrubbed porcelain1)
            ) 
            (not 
                (scrubbed porcelain2)
            )
        ) 
        (inside photograph1 cabinet3) 
        (inside photograph2 cabinet3) 
        (and 
            (dusty photograph1) 
            (dusty photograph2)
        ) 
        (dusty console_table1) 
        (ontop atomizer2 console_table1) 
        (ontop scrub_brush2 console_table1) 
        (under bucket2 console_table1) 
        (inside soap2 cabinet4) 
        (and 
            (inside rag3 cabinet4) 
            (inside rag4 cabinet4)
        ) 
        (and 
            (not 
                (soaked rag3)
            ) 
            (not 
                (soaked rag4)
            )
        ) 
        (inroom console_table1 diningroom) 
        (inroom counter1 bathroom) 
        (inroom cabinet1 bathroom) 
        (inroom cabinet2 bathroom) 
        (inroom cabinet3 diningroom) 
        (inroom cabinet4 diningroom) 
        (inroom sink1 bathroom) 
        (inroom towel_rack1 bathroom)
    )
    
    (:goal 
        (and 
            (forall 
                (?cabinet - cabinet) 
                (scrubbed ?cabinet)
            ) 
            (forall 
                (?cabinet - cabinet) 
                (not 
                    (dusty ?cabinet)
                )
            ) 
            (forall 
                (?rag - rag) 
                (soaked ?rag)
            ) 
            (and 
                (inside ?rag1 ?bucket1) 
                (inside ?rag2 ?bucket1)
            ) 
            (and 
                (inside ?rag3 ?bucket2) 
                (inside ?rag4 ?bucket2)
            ) 
            (forall 
                (?towel - towel) 
                (scrubbed ?towel)
            ) 
            (forn 
                (2) 
                (?towel - towel) 
                (ontop ?towel ?towel_rack1)
            ) 
            (inside ?atomizer1 ?cabinet1) 
            (inside ?soap1 ?cabinet1) 
            (inside ?scrub_brush1 ?cabinet2) 
            (inside ?bucket1 ?cabinet2) 
            (forall 
                (?porcelain - porcelain) 
                (scrubbed ?porcelain)
            ) 
            (inside ?porcelain1 ?cabinet3) 
            (inside ?porcelain2 ?cabinet3) 
            (forall 
                (?photograph - photograph) 
                (not 
                    (dusty ?photograph)
                )
            ) 
            (inside ?photograph1 ?cabinet3) 
            (inside ?photograph2 ?cabinet3) 
            (not 
                (dusty ?console_table1)
            ) 
            (inside ?atomizer2 ?cabinet4) 
            (inside ?scrub_brush2 ?cabinet4) 
            (inside ?soap2 ?cabinet4) 
            (inside ?bucket2 ?cabinet4)
        )
    )
)