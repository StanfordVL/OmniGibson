(define (problem laying_linoleum_floors_0)
    (:domain igibson)

    (:objects
     	cutter1 - cutter
    	cabinet1 - cabinet
    	tool1 - tool
    	laminate1 laminate10 laminate11 laminate12 laminate13 laminate14 laminate15 laminate16 laminate17 laminate18 laminate19 laminate2 laminate20 laminate21 laminate22 laminate23 laminate24 laminate25 laminate26 laminate27 laminate28 laminate29 laminate3 laminate30 laminate4 laminate5 laminate6 laminate7 laminate8 laminate9 - laminate
    )
    
    (:init 
        (inside cutter1 cabinet1) 
        (inside edge_tool1 cabinet1) 
        (inside laminate1 cabinet1) 
        (inside laminate21 cabinet1) 
        (inside laminate22 cabinet1) 
        (inside laminate23 cabinet1) 
        (inside laminate24 cabinet1) 
        (inside laminate25 cabinet1) 
        (inside laminate26 cabinet1) 
        (inside laminate27 cabinet1) 
        (inside laminate28 cabinet1) 
        (inside laminate29 cabinet1) 
        (inside laminate2 cabinet1) 
        (inside laminate30 cabinet1) 
        (inside laminate16 cabinet1) 
        (inside laminate18 cabinet1) 
        (inside laminate19 cabinet1) 
        (inside laminate20 cabinet1) 
        (inside laminate3 cabinet1) 
        (inside laminate4 cabinet1) 
        (inside laminate5 cabinet1) 
        (inside laminate6 cabinet1) 
        (inside laminate7 cabinet1) 
        (inside laminate8 cabinet1) 
        (inside laminate9 cabinet1) 
        (inside laminate10 cabinet1) 
        (inside laminate11 cabinet1) 
        (inside laminate12 cabinet1) 
        (inside laminate17 cabinet1) 
        (inside laminate13 cabinet1) 
        (inside laminate14 cabinet1) 
        (inside laminate15 cabinet1)
    )
    
    (:goal 
        (and 
            (forn 
                (30) 
                (?laminate - laminate) 
                (ontop ?laminate ?carpet)
            ) 
            (forall 
                (?edge_tool - edge_tool) 
                (not 
                    (inside ?edge_tool ?cabinet)
                )
            ) 
            (forall 
                (?cutter - cutter) 
                (not 
                    (inside ?cutter ?cabinet)
                )
            )
        )
    )
)