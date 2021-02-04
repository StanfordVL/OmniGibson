(define (problem organizing_school_stuff_1
    (:domain igibson)

    (:objects
     	tablet1 - tablet
    	shelf1 - shelf
    	pouch1 - pouch
    	table1 - table
    	backpack1 - backpack
    	book1 book2 book3 - book
    	folder1 folder2 - folder
    	pen1 pen2 pen3 pen4 - pen
    	pencil1 pencil2 - pencil
    )
    
    (:init 
        (ontop tablet1 shelf1) 
        (and 
            (ontop pouch1 table1) 
            (open pouch1)
        ) 
        (and 
            (ontop backpack1 table1) 
            (open backpack1)
        ) 
        (and 
            (ontop book1 shelf1) 
            (ontop book2 shelf1) 
            (ontop book3 shelf1) 
            (ontop folder1 shelf1) 
            (ontop folder2 shelf1)
        ) 
        (and 
            (ontop pen1 table1) 
            (ontop pen2 table1) 
            (ontop pen3 table1) 
            (ontop pen4 table1) 
            (ontop pencil1 table1) 
            (ontop pencil2 table1)
        )
    )
    
    (:goal 
        (and 
            (and 
                (forall 
                    (?book - book) 
                    (inside ?book ?backpack1)
                ) 
                (forall 
                    (?folder - folder) 
                    (inside ?folder ?backpack1)
                ) 
                (and 
                    (and 
                        (forall 
                            (?pen - pen) 
                            (inside ?pen ?pouch1)
                        ) 
                        (forall 
                            (?pencil - pencil) 
                            (inside ?pencil ?pouch1)
                        )
                    ) 
                    (not 
                        (open ?pouch1)
                    ) 
                    (inside ?pouch1 ?backpack1)
                ) 
                (inside ?tablet1 ?backpack1)
            ) 
            (not 
                (open ?backpack1)
            )
        )
    )
)