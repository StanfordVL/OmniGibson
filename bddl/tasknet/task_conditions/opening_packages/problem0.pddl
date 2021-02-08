(define (problem opening_packages_0)
    (:domain igibson)

    (:objects
     	scissors1 - scissors
    	table1 - table
    	blade1 - blade
    	knife1 - knife
    	package1 package2 package3 package4 package5 package6 package7 package8 - package
    )
    
    (:init 
        (and 
            (ontop scissors1 table1) 
            (ontop blade1 table1) 
            (ontop knife1 table1)
        ) 
        (and 
            (ontop package1 table1) 
            (not 
                (open package1)
            )
        ) 
        (and 
            (ontop package2 table1) 
            (not 
                (open package2)
            )
        ) 
        (and 
            (ontop package3 table1) 
            (not 
                (open package3)
            )
        ) 
        (and 
            (ontop package4 table1) 
            (not 
                (open package4)
            )
        ) 
        (and 
            (ontop package5 table1) 
            (not 
                (open package5)
            )
        ) 
        (and 
            (ontop package6 table1) 
            (not 
                (open package6)
            )
        ) 
        (and 
            (ontop package7 table1) 
            (not 
                (open package7)
            )
        ) 
        (and 
            (ontop package8 table1) 
            (not 
                (open package8)
            )
        )
    )
    
    (:goal 
        (and 
            (forall 
                (?package - package) 
                (open ?package)
            )
        )
    )
)