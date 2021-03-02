(define (problem cleaning_toilet_0)
    (:domain igibson)

    (:objects
     	bucket1 - bucket
    	floor1 - floor
    	toilet1 - toilet
    	cabinet1 - cabinet
    	scrub_brush1 - scrub_brush
    	squeegee1 - squeegee
    	plunger1 - plunger
    	mold1 mold2 mold3 mold4 - mold
    	soap1 - soap
    	detergent1 - detergent
    )
    
    (:init 
        (ontop bucket1 floor1) 
        (not 
            (scrubbed toilet1)
        ) 
        (ontop toilet1 floor1) 
        (ontop cabinet1 floor1) 
        (nextto scrub_brush1 toilet1) 
        (nextto squeegee1 toilet1) 
        (nextto plunger1 toilet1) 
        (inside mold1 toilet1) 
        (inside mold2 toilet1) 
        (inside mold2 toilet1) 
        (inside mold3 toilet1) 
        (inside mold4 toilet1) 
        (nextto soap1 toilet1) 
        (nextto detergent1 toilet1) 
        (inroom cabinet1 bathroom) 
        (inroom floor1 bathroom) 
        (inroom toilet1 bathroom)
    )
    
    (:goal 
        (and 
            (scrubbed ?toilet1) 
            (forall 
                (?mold - mold) 
                (inside ?mold ?bucket1)
            ) 
            (inside ?scrub_brush1 ?cabinet1) 
            (inside ?squeegee1 ?cabinet1) 
            (inside ?soap1 ?toilet1) 
            (inside ?detergent1 ?toilet1) 
            (inside ?plunger1 ?cabinet1) 
            (ontop ?cabinet1 ?floor1)
        )
    )
)