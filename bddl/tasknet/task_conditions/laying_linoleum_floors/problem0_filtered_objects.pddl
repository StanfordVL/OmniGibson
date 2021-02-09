(define (problem laying_linoleum_floors_0)
    (:domain igibson)

    (:objects
     	carving_knife1 - carving_knife
    	cabinet1 - cabinet
    	screwdriver1 - screwdriver
    	plywood1 plywood10 plywood11 plywood12 plywood13 plywood14 plywood15 plywood16 plywood17 plywood18 plywood19 plywood2 plywood20 plywood21 plywood22 plywood23 plywood24 plywood25 plywood26 plywood27 plywood28 plywood29 plywood3 plywood30 plywood4 plywood5 plywood6 plywood7 plywood8 plywood9 - plywood
    )
    
    (:init 
        (inside carving_knife1 cabinet1) 
        (inside screwdriver1 cabinet1) 
        (inside plywood1 cabinet1) 
        (inside plywood21 cabinet1) 
        (inside plywood22 cabinet1) 
        (inside plywood23 cabinet1) 
        (inside plywood24 cabinet1) 
        (inside plywood25 cabinet1) 
        (inside plywood26 cabinet1) 
        (inside plywood27 cabinet1) 
        (inside plywood28 cabinet1) 
        (inside plywood29 cabinet1) 
        (inside plywood2 cabinet1) 
        (inside plywood30 cabinet1) 
        (inside plywood16 cabinet1) 
        (inside plywood18 cabinet1) 
        (inside plywood19 cabinet1) 
        (inside plywood20 cabinet1) 
        (inside plywood3 cabinet1) 
        (inside plywood4 cabinet1) 
        (inside plywood5 cabinet1) 
        (inside plywood6 cabinet1) 
        (inside plywood7 cabinet1) 
        (inside plywood8 cabinet1) 
        (inside plywood9 cabinet1) 
        (inside plywood10 cabinet1) 
        (inside plywood11 cabinet1) 
        (inside plywood12 cabinet1) 
        (inside plywood17 cabinet1) 
        (inside plywood13 cabinet1) 
        (inside plywood14 cabinet1) 
        (inside plywood15 cabinet1)
    )
    
    (:goal 
        (and 
            (forn 
                (30) 
                (?plywood - plywood) 
                (ontop ?plywood ?carpet)
            ) 
            (forall 
                (?screwdriver - screwdriver) 
                (not 
                    (inside ?screwdriver ?cabinet)
                )
            ) 
            (forall 
                (?carving_knife - carving_knife) 
                (not 
                    (inside ?carving_knife ?cabinet)
                )
            )
        )
    )
)