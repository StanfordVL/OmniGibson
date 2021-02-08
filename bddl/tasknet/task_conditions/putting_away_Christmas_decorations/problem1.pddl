(define (problem putting_away_Christmas_decorations_1)
    (:domain igibson)

    (:objects
     	shelf1 - shelf
    	cabinet1 - cabinet
    	wreath1 wreath2 - wreath
    	bow1 bow2 bow3 bow4 bow5 bow6 - bow
    	carpet1 - carpet
    	ball1 ball2 - ball
    	wrapping1 wrapping2 wrapping3 wrapping4 wrapping5 wrapping6 - wrapping
    )
    
    (:init 
        (nextto shelf1 cabinet1) 
        (ontop wreath1 shelf1) 
        (ontop wreath2 shelf1) 
        (ontop bow1 carpet1) 
        (ontop bow2 carpet1) 
        (ontop bow3 carpet1) 
        (ontop bow4 carpet1) 
        (ontop bow5 carpet1) 
        (ontop bow6 carpet1) 
        (ontop ball1 carpet1) 
        (ontop ball2 carpet1) 
        (ontop wrapping1 carpet1) 
        (ontop wrapping2 carpet1) 
        (ontop wrapping3 carpet1) 
        (ontop wrapping4 carpet1) 
        (ontop wrapping5 carpet1) 
        (ontop wrapping6 carpet1) 
        (nextto carpet1 cabinet1) 
        (inroom shelf1 garage) 
        (inroom cabinet1 garage) 
        (inroom carpet1 garage)
    )
    
    (:goal 
        (and 
            (forall 
                (?ball - ball) 
                (inside ?ball ?cabinet1)
            ) 
            (forall 
                (?wreath - wreath) 
                (inside ?wreath ?cabinet1)
            ) 
            (forall 
                (?wrapping - wrapping) 
                (inside ?wrapping ?cabinet1)
            ) 
            (forall 
                (?bow - bow) 
                (inside ?bow ?cabinet1)
            )
        )
    )
)