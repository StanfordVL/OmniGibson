(define (problem cleaning_bedroom_0)
    (:domain igibson)

    (:objects
     	carpet1 - carpet
            cabinet1 - cabinet
    	table1 - table
    	office_chair1 - office_chair
    	door1 - door
    	hook1 - hook
    	dustcloth1 - dustcloth
    	vacuum1 - vacuum
    	window1 - window
    	bed1 - bed
    	shirt1 shirt2 - shirt
    	coat1 - coat
    	skirt1 - skirt
    	bed_linen1 - bed_linen
    	pillow1 - pillow
    	sock1 sock2 sock3 sock4 - sock
    	underwear1 underwear2 - underwear
    	hamper1 - hamper
    	laundry1 laundry2 laundry3 - laundry
    	notebook1 notebook2 - notebook
    	pen1 pen2 - pen
    	backpack1 - backpack
    	gym_shoe1 gym_shoe2 - gym_shoe
    )
    
    (:init 
        (and 
            (dusty carpet1) 
            (not 
                (scrubbed carpet1)
            ) 
            (dusty table1) 
            (nextto office_chair1 table1) 
            (open door1) 
            (ontop hook1 door1) 
            (ontop dustcloth1 table1) 
            (nextto vacuum1 table1) 
            (and 
                (nextto window1 table1) 
                (not 
                    (dusty window1)
                ) 
                (not 
                    (open window1)
                )
            )
        ) 
        (and 
            (ontop shirt1 bed1) 
            (ontop shirt2 bed1) 
            (ontop coat1 bed1) 
            (ontop skirt1 bed1) 
            (ontop bed_linen1 office_chair1) 
            (ontop pillow1 office_chair1)
        ) 
        (and 
            (ontop sock1 carpet1) 
            (ontop sock2 carpet1) 
            (ontop sock3 carpet1) 
            (ontop sock4 carpet1) 
            (ontop underwear1 carpet1) 
            (ontop underwear2 carpet1)
        ) 
        (and 
            (ontop hamper1 carpet1) 
            (nextto laundry1 bed1) 
            (inside laundry2 bed1) 
            (nextto laundry3 bed1)
        ) 
        (and 
            (nextto notebook1 bed1) 
            (nextto notebook2 table1) 
            (nextto pen1 bed1) 
            (nextto pen2 table1) 
            (nextto backpack1 table1) 
            (ontop gym_shoe1 carpet1) 
            (ontop gym_shoe2 carpet1)
        ) 
        (inroom carpet1 bedroom) 
        (inroom bed1 bedroom) 
        (inroom window1 bedroom) 
        (inroom table1 bedroom) 
        (inroom office_chair1 bedroom) 
        (inroom door1 bedroom) 
        (inroom cabinet1 bedroom)
    )
    
    (:goal 
        (and 
            (and 
                (scrubbed ?carpet1) 
                (not 
                    (dusty ?carpet1)
                ) 
                (and 
                    (ontop ?bed_linen1 ?bed1) 
                    (ontop ?pillow1 ?bed1)
                ) 
                (open ?window1) 
                (not 
                    (dusty ?window1)
                ) 
                (not 
                    (dusty ?table1)
                ) 
                (under ?office_chair1 ?table1) 
                (and 
                    (open ?door1) 
                    (ontop ?hook1 ?door1) 
                    (ontop ?coat1 ?hook1)
                )
            ) 
            (forall 
                (?laundry - laundry) 
                (inside ?laundry ?hamper1)
            ) 
            (forall 
                (?shirt - shirt) 
                (inside ?shirt ?cabinet1)
            ) 
            (forall 
                (?sock - sock) 
                (inside ?sock ?cabinet1)
            ) 
            (forall 
                (?underwear - underwear) 
                (inside ?underwear ?cabinet1)
            ) 
            (inside ?skirt1 ?cabinet1) 
            (and 
                (fornpairs 
                    (2) 
                    (?notebook - notebook) 
                    (?pen - pen) 
                    (and 
                        (inside ?notebook ?backpack1) 
                        (inside ?pen ?backpack1)
                    )
                ) 
                (ontop ?backpack1 ?table1)
            ) 
            (forall 
                (?gym_shoe - gym_shoe) 
                (under ?gym_shoe ?bed1)
            ) 
            (and 
                (nextto ?vacuum1 ?table1) 
                (ontop ?dustcloth1 ?table1)
            )
        )
    )
)