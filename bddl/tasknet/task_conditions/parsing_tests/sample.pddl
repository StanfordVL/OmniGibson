(define (problem serving_a_meal) ()

    (:domain igibson)
    (:objects
        plate1 plate2 - plate
        cabinet1 - cabinet
        bowl1 bowl2 - bowl
        fork1 fork2 - fork
        knife1 knife2 - knife
        spoon1 spoon2 - spoon
        gazpacho1 gazpacho2 - gazpacho
        fridge1 - fridge
        tenderloin1 tenderloin2 - tenderloin
        stove1 - stove
        chair1 chair2 - chair
        floor1 - floor
        table1 - table
    )
    
    (:init 
        (inside plate1 cabinet1) 
        (inside plate2 cabinet1) 
        (inside bowl1 cabinet1) 
        (inside bowl2 cabinet1) 
        (inside fork1 cabinet1) 
        (inside fork2 cabinet1) 
        (inside knife1 cabinet1) 
        (inside knife2 cabinet1) 
        (inside spoon1 cabinet1) 
        (inside spoon2 cabinet1) 
        (inside gazpacho1 fridge1) 
        (inside gazpacho2 fridge1) 
        (ontop tenderloin1 stove1) 
        (ontop tenderloin2 stove1) 
        (inroom stove1 kitchen) 
        (inroom chair1 dining_room) 
        (inroom chair2 dining_room) 
        (inroom floor1 dining_room) 
        (inroom table1 dining_room) 
        (inroom cabinet1 kitchen) 
        (inroom fridge1 kitchen)
    )
)


(:goal 
    (and 
        (forpairs 
            (?plate - plate) (?chair - chair) 
            (and (ontop ?plate ?table1) (nextto ?plate ?chair))
        ) 
        (forpairs 
            (?bowl - bowl) (?chair - chair) 
            (and (ontop ?bowl ?table1) (nextto ?bowl ?chair))
        ) 
        (forpairs 
            (?fork - fork) (?plate - plate) 
            (and (ontop ?fork ?table1) (nextto ?fork ?plate))
        )  
        (forpairs 
            (?knife - knife) (?plate - plate) 
            (and (ontop ?knife ?table1) (nextto ?knife ?plate))
        ) 
        (forpairs 
            (?spoon - spoon) (?plate - plate) 
            (and (ontop ?spoon ?table1) (nextto ?spoon ?plate))
        ) 
        (forall 
            (?tenderloin - tenderloin) 
            (cooked ?tenderloin)
        ) 
        (forpairs 
            (?tenderloin - tenderloin) (?plate - plate) 
            (ontop ?tenderloin ?plate)
        ) 
        (forall 
            (?gazpacho - gazpacho) 
            (not (frozen ?gazpacho))
        ) 
        (forpairs 
            (?gazpacho - gazpacho) (?bowl - bowl) 
            (inside ?gazpacho ?bowl)
        )
    )
)