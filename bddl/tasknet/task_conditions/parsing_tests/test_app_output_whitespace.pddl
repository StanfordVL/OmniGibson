
(:goal 
    (and 
        (forall 
            (?basket - basket) 
            (forn 
                (2) 
                (?egg - egg) 
                (inside ?egg ?basket)
            )
        ) 
        (forpairs 
            (?chocolate - chocolate) 
            (?basket - basket) 
            (inside ?chocolate ?basket)
        ) 
        (forpairs 
            (?coloring_material - coloring_material) 
            (?basket - basket) 
            (inside ?coloring_material ?basket)
        ) 
        (forall 
            (?basket - basket) 
            (forn 
                (2) 
                (?crayon - crayon) 
                (inside ?crayon ?basket)
            )
        )
    )
)