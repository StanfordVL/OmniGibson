(define (problem decorating_the_Christmas_tree_1)
    (:domain igibson)

    (:objects
     	tree1 - tree
    	floor1 - floor
    	bell1 bell2 bell3 - bell
    	table1 - table
    	berry1 berry2 berry3 berry4 berry5 - berry
    	light_bulb1 light_bulb2 light_bulb3 light_bulb4 light_bulb5 light_bulb6 - light_bulb
    	bead1 bead2 - bead
    )
    
    (:init 
        (ontop tree1 floor1) 
        (ontop bell1 table1) 
        (ontop bell2 table1) 
        (ontop bell3 table1) 
        (ontop berry1 table1) 
        (ontop berry2 table1) 
        (ontop berry3 table1) 
        (ontop berry4 table1) 
        (ontop berry5 table1) 
        (ontop light_bulb1 table1) 
        (ontop light_bulb2 table1) 
        (ontop light_bulb3 table1) 
        (ontop light_bulb4 table1) 
        (ontop light_bulb5 table1) 
        (ontop light_bulb6 table1) 
        (ontop bead1 table1) 
        (ontop bead2 table1) 
        (inroom floor1 living room) 
        (inroom table1 living room)
    )
    
    (:goal 
        (and 
            (forall 
                (?bell - bell) 
                (ontop ?bell ?tree1)
            ) 
            (forall 
                (?berry - berry) 
                (ontop ?berry ?tree1)
            ) 
            (forall 
                (?light_bulb - light_bulb) 
                (ontop ?light_bulb ?tree1)
            ) 
            (forall 
                (?bead - bead) 
                (ontop ?bead ?tree1)
            )
        )
    )
)