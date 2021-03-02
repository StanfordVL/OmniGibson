(define (problem washing_pots_and_pans_0)
    (:domain igibson)

    (:objects
     	sink1 - sink
    	counter1 - counter
    	faucet1 - faucet
    	water1 - water
    	drain1 - drain
    	scrub_brush1 - scrub_brush
    	towel1 - towel
    	pan1 pan2 pan3 pan4 - pan
    	vegetable_oil1 vegetable_oil2 - vegetable_oil
    	pot1 pot2 - pot
    	juice1 - juice
    	cabinet1 - cabinet
    	soap1 - soap
    	vessel1 - vessel
    )
    
    (:init 
        (and 
            (ontop sink1 counter1) 
            (nextto faucet1 sink1) 
            (and 
                (inside water1 faucet1) 
                (nextto sink1 water1)
            ) 
            (under drain1 sink1) 
            (ontop scrub_brush1 counter1) 
            (ontop towel1 counter1)
        ) 
        (and 
            (inside pan1 sink1) 
            (dusty pan1) 
            (not 
                (scrubbed pan1)
            ) 
            (inside vegetable_oil1 pan1) 
            (under sink1 vegetable_oil1)
        ) 
        (and 
            (inside pan2 sink1) 
            (dusty pan2) 
            (not 
                (scrubbed pan2)
            ) 
            (inside vegetable_oil2 pan2) 
            (under sink1 vegetable_oil2)
        ) 
        (and 
            (inside pot1 sink1) 
            (dusty pot1) 
            (not 
                (scrubbed pot1)
            ) 
            (ontop juice1 pot1) 
            (under sink1 juice1)
        ) 
        (and 
            (ontop pan3 counter1) 
            (dusty pan3) 
            (not 
                (scrubbed pan3)
            )
        ) 
        (and 
            (ontop pan4 counter1) 
            (dusty pan4) 
            (not 
                (scrubbed pan4)
            )
        ) 
        (and 
            (ontop pot2 counter1) 
            (dusty pot2) 
            (not 
                (scrubbed pot2)
            )
        ) 
        (nextto cabinet1 sink1) 
        (and 
            (inside soap1 vessel1) 
            (ontop vessel1 counter1) 
            (under counter1 soap1)
        ) 
        (inroom sink1 kitchen) 
        (inroom counter1 kitchen) 
        (inroom cabinet1 kitchen)
    )
    
    (:goal 
        (and 
            (forall 
                (?pan - pan) 
                (and 
                    (scrubbed ?pan) 
                    (not 
                        (dusty ?pan)
                    ) 
                    (inside ?pan ?cabinet1)
                )
            ) 
            (forall 
                (?pot - pot) 
                (and 
                    (scrubbed ?pot) 
                    (not 
                        (dusty ?pot)
                    ) 
                    (inside ?pot ?cabinet1)
                )
            ) 
            (and 
                (forall 
                    (?vegetable_oil - vegetable_oil) 
                    (inside ?vegetable_oil ?drain1)
                ) 
                (inside ?juice1 ?drain1) 
                (inside ?water1 ?drain1) 
                (inside ?soap1 ?drain1)
            ) 
            (and 
                (ontop ?scrub_brush1 ?counter1) 
                (soaked ?scrub_brush1)
            ) 
            (and 
                (ontop ?towel1 ?counter1) 
                (ontop ?vessel1 ?counter1) 
                (not 
                    (inside ?soap1 ?vessel1)
                )
            )
        )
    )
)