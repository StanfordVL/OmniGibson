(define (problem laying_linoleum_floors_0)
    (:domain igibson)

    (:objects
     	carving_knife1 - carving_knife
    	top_cabinet1 - top_cabinet
    	screwdriver1 - screwdriver
    	plywood1 plywood10 plywood11 plywood12 plywood13 plywood14 plywood15 plywood16 plywood17 plywood18 plywood19 plywood2 plywood20 plywood21 plywood22 plywood23 plywood24 plywood25 plywood26 plywood27 plywood28 plywood29 plywood3 plywood30 plywood4 plywood5 plywood6 plywood7 plywood8 plywood9 - plywood
    )
    
    (:init 
        (inside carving_knife1 top_cabinet1) 
        (inside screwdriver1 top_cabinet1) 
        (inside plywood1 top_cabinet1) 
        (inside plywood21 top_cabinet1) 
        (inside plywood22 top_cabinet1) 
        (inside plywood23 top_cabinet1) 
        (inside plywood24 top_cabinet1) 
        (inside plywood25 top_cabinet1) 
        (inside plywood26 top_cabinet1) 
        (inside plywood27 top_cabinet1) 
        (inside plywood28 top_cabinet1) 
        (inside plywood29 top_cabinet1) 
        (inside plywood2 top_cabinet1) 
        (inside plywood30 top_cabinet1) 
        (inside plywood16 top_cabinet1) 
        (inside plywood18 top_cabinet1) 
        (inside plywood19 top_cabinet1) 
        (inside plywood20 top_cabinet1) 
        (inside plywood3 top_cabinet1) 
        (inside plywood4 top_cabinet1) 
        (inside plywood5 top_cabinet1) 
        (inside plywood6 top_cabinet1) 
        (inside plywood7 top_cabinet1) 
        (inside plywood8 top_cabinet1) 
        (inside plywood9 top_cabinet1) 
        (inside plywood10 top_cabinet1) 
        (inside plywood11 top_cabinet1) 
        (inside plywood12 top_cabinet1) 
        (inside plywood17 top_cabinet1) 
        (inside plywood13 top_cabinet1) 
        (inside plywood14 top_cabinet1) 
        (inside plywood15 top_cabinet1)
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
                    (inside ?screwdriver ?top_cabinet)
                )
            ) 
            (forall 
                (?carving_knife - carving_knife) 
                (not 
                    (inside ?carving_knife ?top_cabinet)
                )
            )
        )
    )
)
