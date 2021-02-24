(define (problem packing_lunches_0)
    (:domain igibson)

    (:objects
     	shelf.n.01_1 - shelf.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	bag.n.01_1 bag.n.01_2 bag.n.01_3 bag.n.01_4 - bag.n.01
    	yogurt.n.01_1 yogurt.n.01_2 yogurt.n.01_3 yogurt.n.01_4 - yogurt.n.01
    	hamburger.n.01_1 hamburger.n.01_2 hamburger.n.01_3 hamburger.n.01_4 - hamburger.n.01
    	melon.n.01_1 melon.n.01_2 melon.n.01_3 melon.n.01_4 - melon.n.01
    	pop.n.02_1 pop.n.02_2 pop.n.02_3 pop.n.02_4 - pop.n.02
    )
    
    (:init 
        (nextto shelf.n.01_1 electric_refrigerator.n.01_1) 
        (ontop bag.n.01_1 shelf.n.01_1) 
        (ontop bag.n.01_2 shelf.n.01_1) 
        (ontop bag.n.01_3 shelf.n.01_1) 
        (ontop bag.n.01_4 shelf.n.01_1) 
        (cooked yogurt.n.01_1) 
        (inside yogurt.n.01_1 electric_refrigerator.n.01_1) 
        (cooked yogurt.n.01_2) 
        (inside yogurt.n.01_2 electric_refrigerator.n.01_1) 
        (cooked yogurt.n.01_3) 
        (inside yogurt.n.01_3 electric_refrigerator.n.01_1) 
        (cooked yogurt.n.01_4) 
        (inside yogurt.n.01_4 electric_refrigerator.n.01_1) 
        (ontop hamburger.n.01_1 shelf.n.01_1) 
        (ontop hamburger.n.01_2 shelf.n.01_1) 
        (ontop hamburger.n.01_3 shelf.n.01_1) 
        (ontop hamburger.n.01_4 shelf.n.01_1) 
        (inside melon.n.01_1 electric_refrigerator.n.01_1) 
        (inside melon.n.01_2 electric_refrigerator.n.01_1) 
        (inside melon.n.01_3 electric_refrigerator.n.01_1) 
        (inside melon.n.01_4 electric_refrigerator.n.01_1) 
        (inside pop.n.02_1 electric_refrigerator.n.01_1) 
        (inside pop.n.02_2 electric_refrigerator.n.01_1) 
        (inside pop.n.02_3 electric_refrigerator.n.01_1) 
        (inside pop.n.02_4 electric_refrigerator.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
            (nextto ?shelf.n.01_1 ?electric_refrigerator.n.01_1) 
            (forpairs 
                (?melon.n.01 - melon.n.01) 
                (?bag.n.01 - bag.n.01) 
                (inside ?melon.n.01 ?bag.n.01)
            ) 
            (forpairs 
                (?yogurt.n.01 - yogurt.n.01) 
                (?bag.n.01 - bag.n.01) 
                (inside ?yogurt.n.01 ?bag.n.01)
            ) 
            (forpairs 
                (?hamburger.n.01 - hamburger.n.01) 
                (?bag.n.01 - bag.n.01) 
                (inside ?hamburger.n.01 ?bag.n.01)
            ) 
            (forpairs 
                (?pop.n.02 - pop.n.02) 
                (?bag.n.01 - bag.n.01) 
                (inside ?pop.n.02 ?bag.n.01)
            )
        )
    )
)
