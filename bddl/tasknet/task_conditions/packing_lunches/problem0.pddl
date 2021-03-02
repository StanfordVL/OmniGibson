(define (problem packing_lunches_0)
    (:domain igibson)

    (:objects
     	shelf1 - shelf
    	fridge1 - fridge
    	bag1 bag2 bag3 bag4 - bag
    	tuna1 tuna2 tuna3 tuna4 - tuna
    	pita1 pita2 pita3 pita4 - pita
    	melon1 melon2 melon3 melon4 - melon
    	lemonade1 lemonade2 lemonade3 lemonade4 - lemonade
    )
    
    (:init 
        (nextto shelf1 fridge1) 
        (ontop bag1 shelf1) 
        (ontop bag2 shelf1) 
        (ontop bag3 shelf1) 
        (ontop bag4 shelf1) 
        (cooked tuna1) 
        (inside tuna1 fridge1) 
        (cooked tuna2) 
        (inside tuna2 fridge1) 
        (cooked tuna3) 
        (inside tuna3 fridge1) 
        (cooked tuna4) 
        (inside tuna4 fridge1) 
        (ontop pita1 shelf1) 
        (ontop pita2 shelf1) 
        (ontop pita3 shelf1) 
        (ontop pita4 shelf1) 
        (inside melon1 fridge1) 
        (inside melon2 fridge1) 
        (inside melon3 fridge1) 
        (inside melon4 fridge1) 
        (inside lemonade1 fridge1) 
        (inside lemonade2 fridge1) 
        (inside lemonade3 fridge1) 
        (inside lemonade4 fridge1) 
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
                (?tuna - tuna) 
                (?bag - bag) 
                (inside ?tuna ?bag)
            ) 
            (forpairs 
                (?pita - pita) 
                (?bag - bag) 
                (inside ?pita ?bag)
            ) 
            (forpairs 
                (?lemonade - lemonade) 
                (?bag - bag) 
                (inside ?lemonade ?bag)
            )
        )
    )
)