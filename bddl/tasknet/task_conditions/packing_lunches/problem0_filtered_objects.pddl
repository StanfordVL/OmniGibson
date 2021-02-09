(define (problem packing_lunches_0)
    (:domain igibson)

    (:objects
     	shelf1 - shelf
    	fridge1 - fridge
    	bag1 bag2 bag3 bag4 - bag
    	yogurt1 yogurt2 yogurt3 yogurt4 - yogurt
    	hamburger1 hamburger2 hamburger3 hamburger4 - hamburger
    	melon1 melon2 melon3 melon4 - melon
    	soda1 soda2 soda3 soda4 - soda
    )
    
    (:init 
        (nextto shelf1 fridge1) 
        (ontop bag1 shelf1) 
        (ontop bag2 shelf1) 
        (ontop bag3 shelf1) 
        (ontop bag4 shelf1) 
        (cooked yogurt1) 
        (inside yogurt1 fridge1) 
        (cooked yogurt2) 
        (inside yogurt2 fridge1) 
        (cooked yogurt3) 
        (inside yogurt3 fridge1) 
        (cooked yogurt4) 
        (inside yogurt4 fridge1) 
        (ontop hamburger1 shelf1) 
        (ontop hamburger2 shelf1) 
        (ontop hamburger3 shelf1) 
        (ontop hamburger4 shelf1) 
        (inside melon1 fridge1) 
        (inside melon2 fridge1) 
        (inside melon3 fridge1) 
        (inside melon4 fridge1) 
        (inside soda1 fridge1) 
        (inside soda2 fridge1) 
        (inside soda3 fridge1) 
        (inside soda4 fridge1) 
        (inroom fridge1 kitchen) 
        (inroom shelf1 kitchen)
    )
    
    (:goal 
        (and 
            (nextto ?shelf1 ?fridge1) 
            (forpairs 
                (?melon - melon) 
                (?bag - bag) 
                (inside ?melon ?bag)
            ) 
            (forpairs 
                (?yogurt - yogurt) 
                (?bag - bag) 
                (inside ?yogurt ?bag)
            ) 
            (forpairs 
                (?hamburger - hamburger) 
                (?bag - bag) 
                (inside ?hamburger ?bag)
            ) 
            (forpairs 
                (?soda - soda) 
                (?bag - bag) 
                (inside ?soda ?bag)
            )
        )
    )
)
