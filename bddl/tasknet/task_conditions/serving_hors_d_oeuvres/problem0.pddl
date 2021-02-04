(define (problem serving_hors_d_oeuvres_0
    (:domain igibson)

    (:objects
     	snack_food1 snack_food10 snack_food11 snack_food12 snack_food13 snack_food14 snack_food15 snack_food16 snack_food17 snack_food18 snack_food19 snack_food2 snack_food20 snack_food21 snack_food22 snack_food23 snack_food24 snack_food3 snack_food4 snack_food5 snack_food6 snack_food7 snack_food8 snack_food9 - snack_food
    	fridge1 - fridge
    	tray1 tray2 tray3 tray4 - tray
    	counter1 - counter
    	plate1 plate10 plate11 plate12 plate13 plate14 plate15 plate16 plate17 plate18 plate19 plate2 plate20 plate21 plate22 plate23 plate24 plate3 plate4 plate5 plate6 plate7 plate8 plate9 - plate
    	cabinet1 - cabinet
    	table1 table2 - table
    )
    
    (:init 
        (inside snack_food1 fridge1) 
        (inside snack_food2 fridge1) 
        (inside snack_food3 fridge1) 
        (inside snack_food4 fridge1) 
        (inside snack_food5 fridge1) 
        (inside snack_food6 fridge1) 
        (inside snack_food7 fridge1) 
        (inside snack_food8 fridge1) 
        (inside snack_food9 fridge1) 
        (inside snack_food10 fridge1) 
        (inside snack_food11 fridge1) 
        (inside snack_food12 fridge1) 
        (inside snack_food13 fridge1) 
        (inside snack_food14 fridge1) 
        (inside snack_food15 fridge1) 
        (inside snack_food16 fridge1) 
        (inside snack_food17 fridge1) 
        (inside snack_food18 fridge1) 
        (inside snack_food19 fridge1) 
        (inside snack_food20 fridge1) 
        (inside snack_food21 fridge1) 
        (inside snack_food22 fridge1) 
        (inside snack_food23 fridge1) 
        (inside snack_food24 fridge1) 
        (ontop tray1 counter1) 
        (ontop tray2 counter1) 
        (ontop tray3 counter1) 
        (ontop tray4 counter1) 
        (inside plate1 cabinet1) 
        (inside plate2 cabinet1) 
        (inside plate3 cabinet1) 
        (inside plate4 cabinet1) 
        (inside plate5 cabinet1) 
        (inside plate6 cabinet1) 
        (inside plate7 cabinet1) 
        (inside plate8 cabinet1) 
        (inside plate9 cabinet1) 
        (inside plate10 cabinet1) 
        (inside plate11 cabinet1) 
        (inside plate12 cabinet1) 
        (inside plate13 cabinet1) 
        (inside plate14 cabinet1) 
        (inside plate15 cabinet1) 
        (inside plate16 cabinet1) 
        (inside plate17 cabinet1) 
        (inside plate18 cabinet1) 
        (inside plate19 cabinet1) 
        (inside plate20 cabinet1) 
        (inside plate21 cabinet1) 
        (inside plate22 cabinet1) 
        (inside plate23 cabinet1) 
        (inside plate24 cabinet1) 
        (inroom counter1 kitchen) 
        (inroom fridge1 kitchen) 
        (inroom table1 livingroom) 
        (inroom table2 livingroom) 
        (inroom cabinet1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?tray - tray) 
                (fornpairs 
                    (6) 
                    (?snack_food - snack_food) 
                    (?plate - plate) 
                    (and 
                        (ontop ?snack_food ?plate) 
                        (ontop ?plate ?tray)
                    )
                )
            ) 
            (forpairs 
                (?tray - tray) 
                (?table - table) 
                (ontop ?tray ?table)
            )
        )
    )
)