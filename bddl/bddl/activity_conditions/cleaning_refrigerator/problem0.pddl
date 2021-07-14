(define (problem cleaning_refrigerator_0)
    (:domain igibson)

    (:objects
     	fridge1 - fridge
    	counter1 - counter
    	receptacle1 - receptacle
    	fruit1 fruit2 fruit3 fruit4 fruit5 - fruit
    	mold1 mold2 mold3 - mold
    	condiment1 condiment2 - condiment
    	tray1 - tray
    	dessert1 - dessert
    	vessel1 - vessel
    	soap1 - soap
    	water1 - water
    	rag1 - rag
    	brush1 - brush
    )
    
    (:init 
        (and 
            (nextto fridge1 counter1) 
            (open fridge1) 
            (dusty fridge1) 
            (not 
                (scrubbed fridge1)
            ) 
            (nextto receptacle1 fridge1)
        ) 
        (and 
            (and 
                (inside fruit1 fridge1) 
                (ontop mold1 fruit1) 
                (under fridge1 mold1)
            ) 
            (and 
                (inside fruit2 fridge1) 
                (ontop mold2 fruit2) 
                (under fridge1 mold2)
            ) 
            (and 
                (inside fruit3 fridge1) 
                (ontop mold3 fruit3) 
                (under fridge1 mold3)
            ) 
            (and 
                (inside fruit4 fridge1) 
                (inside fruit5 fridge1) 
                (inside condiment1 fridge1) 
                (inside condiment2 fridge1) 
                (ontop tray1 counter1) 
                (ontop dessert1 counter1)
            ) 
            (and 
                (ontop vessel1 counter1) 
                (inside soap1 vessel1) 
                (under counter1 soap1) 
                (inside water1 vessel1) 
                (under counter1 water1)
            ) 
            (ontop rag1 counter1) 
            (not 
                (soaked rag1)
            ) 
            (ontop brush1 counter1)
        ) 
        (inroom fridge1 kitchen) 
        (inroom counter1 kitchen)
    )
    
    (:goal 
        (and 
            (and 
                (scrubbed ?fridge1) 
                (not 
                    (open ?fridge1)
                ) 
                (not 
                    (dusty ?fridge1)
                )
            ) 
            (forall 
                (?fruit - fruit) 
                (and 
                    (imply 
                        (exists 
                            (?mold - mold) 
                            (ontop ?mold ?fruit)
                        ) 
                        (inside ?fruit ?receptacle1)
                    ) 
                    (imply 
                        (not 
                            (exists 
                                (?mold - mold) 
                                (ontop ?mold ?fruit)
                            )
                        ) 
                        (inside ?fruit ?fridge1)
                    )
                )
            ) 
            (and 
                (and 
                    (ontop ?rag1 ?counter1) 
                    (soaked ?rag1)
                ) 
                (inside ?brush1 ?vessel1) 
                (inside ?water1 ?vessel1) 
                (inside ?soap1 ?vessel1)
            ) 
            (and 
                (forall 
                    (?condiment - condiment) 
                    (inside ?condiment ?fridge1)
                ) 
                (inside ?tray1 ?fridge1) 
                (inside ?dessert1 ?fridge1)
            )
        )
    )
)