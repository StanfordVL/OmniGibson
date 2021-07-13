(define (problem opening_packages_1)
    (:domain igibson)

    (:objects
     	package1 package2 package3 package4 package5 package6 package7 package8 - package
    	counter1 - counter
    	knife1 - knife
    	scissors1 - scissors
    	blade1 - blade
    )
    
    (:init 
        (ontop package1 counter1) 
        (ontop package2 counter1) 
        (ontop package3 counter1) 
        (ontop package4 counter1) 
        (ontop package5 counter1) 
        (ontop package6 counter1) 
        (ontop package7 counter1) 
        (ontop package8 counter1) 
        (not 
            (open package1)
        ) 
        (not 
            (open package2)
        ) 
        (not 
            (open package3)
        ) 
        (not 
            (open package4)
        ) 
        (not 
            (open package5)
        ) 
        (not 
            (open package6)
        ) 
        (not 
            (open package7)
        ) 
        (not 
            (open package8)
        ) 
        (ontop knife1 counter1) 
        (ontop scissors1 counter1) 
        (ontop blade1 counter1) 
        (inroom counter1 kitchen)
    )
    
    (:goal 
        (and 
            (forn 
                (8) 
                (?package - package) 
                (open ?package)
            ) 
            (forall 
                (?knife - knife) 
                (ontop ?knife ?counter1)
            ) 
            (forall 
                (?scissors - scissors) 
                (ontop ?scissors ?counter1)
            ) 
            (forall 
                (?blade - blade) 
                (ontop ?blade ?counter1)
            )
        )
    )
)