(define (problem cleaning_closet_0)
    (:domain igibson)

    (:objects
     	vacuum1 - vacuum
    	chest1 - chest
    	jean1 jean2 - jean
    	carpet1 - carpet
    	shirt1 shirt2 - shirt
    	shelf1 shelf2 shelf3 - shelf
    	shoe1 shoe2 - shoe
    	door1 - door
    	jewelry1 jewelry2 jewelry3 - jewelry
    	pajama1 - pajama
    	belt1 - belt
    	dustcloth1 - dustcloth
    	sweatshirt1 sweatshirt2 sweatshirt3 - sweatshirt
    	hook1 - hook
    	robe1 - robe
    	sock1 sock2 - sock
    	mirror1 - mirror
    	underwear1 underwear2 underwear3 - underwear
    )
    
    (:init 
        (nextto vacuum1 chest1) 
        (and 
            (ontop jean1 carpet1) 
            (ontop jean2 carpet1)
        ) 
        (or 
            (not 
                (ontop shirt1 shelf1)
            ) 
            (not 
                (ontop shirt2 shelf1)
            )
        ) 
        (or 
            (ontop shoe1 shelf3) 
            (nextto shoe2 door1)
        ) 
        (or 
            (nextto jewelry1 chest1) 
            (nextto jewelry2 chest1) 
            (under jewelry3 chest1)
        ) 
        (not 
            (ontop pajama1 shelf3)
        ) 
        (ontop belt1 carpet1) 
        (ontop dustcloth1 shelf1) 
        (or 
            (ontop sweatshirt1 carpet1) 
            (ontop sweatshirt2 carpet1) 
            (ontop sweatshirt3 carpet1)
        ) 
        (toggled on vacuum1) 
        (open door1) 
        (ontop hook1 door1) 
        (ontop robe1 carpet1) 
        (or 
            (ontop sock1 shelf1) 
            (ontop sock2 shelf2)
        ) 
        (dusty carpet1) 
        (open chest1) 
        (dusty mirror1) 
        (or 
            (ontop underwear1 chest1) 
            (ontop underwear2 chest1) 
            (ontop underwear3 chest1)
        ) 
        (inroom chest1 bedroom) 
        (inroom mirror1 bedroom) 
        (inroom shelf1 bedroom) 
        (inroom shelf2 bedroom) 
        (inroom shelf3 bedroom) 
        (inroom carpet1 bedroom) 
        (inroom door1 bedroom)
    )
    
    (:goal 
        (and 
            (not 
                (dusty ?carpet1)
            ) 
            (not 
                (open ?chest1)
            ) 
            (not 
                (dusty ?mirror1)
            ) 
            (exists 
                (?chest - chest) 
                (forall 
                    (?underwear - underwear) 
                    (inside ?underwear ?chest)
                )
            ) 
            (exists 
                (?chest - chest) 
                (forall 
                    (?sock - sock) 
                    (inside ?sock ?chest)
                )
            ) 
            (ontop ?shoe1 ?carpet1) 
            (ontop ?shoe2 ?carpet1) 
            (and 
                (ontop ?belt1 ?hook1) 
                (ontop ?robe1 ?hook1)
            ) 
            (ontop ?hook1 ?door1) 
            (exists 
                (?chest - chest) 
                (forall 
                    (?jewelry - jewelry) 
                    (inside ?jewelry ?chest)
                )
            ) 
            (not 
                (open ?door1)
            ) 
            (ontop ?dustcloth1 ?chest1) 
            (not 
                (toggled ?on ?vacuum1)
            ) 
            (inside ?pajama1 ?chest1) 
            (exists 
                (?shelf - shelf) 
                (and 
                    (forall 
                        (?sweatshirt - sweatshirt) 
                        (ontop ?sweatshirt ?shelf)
                    ) 
                    (not 
                        (forall 
                            (?jean - jean) 
                            (ontop ?jean ?shelf)
                        )
                    ) 
                    (not 
                        (forall 
                            (?shirt - shirt) 
                            (ontop ?shirt ?shelf)
                        )
                    )
                )
            ) 
            (exists 
                (?shelf - shelf) 
                (and 
                    (forall 
                        (?jean - jean) 
                        (ontop ?jean ?shelf)
                    ) 
                    (not 
                        (forall 
                            (?shirt - shirt) 
                            (ontop ?shirt ?shelf)
                        )
                    ) 
                    (not 
                        (forall 
                            (?sweatshirt - sweatshirt) 
                            (ontop ?sweatshirt ?shelf)
                        )
                    )
                )
            ) 
            (exists 
                (?shelf - shelf) 
                (and 
                    (forall 
                        (?shirt - shirt) 
                        (ontop ?shirt ?shelf)
                    ) 
                    (not 
                        (forall 
                            (?jean - jean) 
                            (ontop ?jean ?shelf)
                        )
                    ) 
                    (not 
                        (forall 
                            (?sweatshirt - sweatshirt) 
                            (ontop ?sweatshirt ?shelf)
                        )
                    )
                )
            )
        )
    )
)