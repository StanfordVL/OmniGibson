(define (problem packing_lunches_0)
    (:domain igibson)

    (:objects
     	shelf.n.01_1 - shelf.n.01
    	electric_refrigerator.n.01_1 - electric_refrigerator.n.01
    	bag.n.01_1 bag.n.01_2 - bag.n.01
    	yogurt.n.01_1 - yogurt.n.01
    	hamburger.n.01_1 hamburger.n.01_2 - hamburger.n.01
    	melon.n.01_1 - melon.n.01
    	pop.n.02_1 - pop.n.02
    )
    
    (:init 
        (inside bag.n.01_1 shelf.n.01_1) 
        (inside bag.n.01_2 shelf.n.01_1) 
        (inside yogurt.n.01_1 electric_refrigerator.n.01_1) 
        (inside hamburger.n.01_1 shelf.n.01_1) 
        (inside hamburger.n.01_2 shelf.n.01_1) 
        (inside melon.n.01_1 electric_refrigerator.n.01_1) 
        (inside pop.n.02_1 electric_refrigerator.n.01_1) 
        (inroom electric_refrigerator.n.01_1 kitchen) 
        (inroom shelf.n.01_1 kitchen)
    )
    
    (:goal 
        (and 
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
