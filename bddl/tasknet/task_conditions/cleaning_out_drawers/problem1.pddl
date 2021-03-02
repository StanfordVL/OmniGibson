(define (problem cleaning_out_drawers_1)
    (:domain igibson)

    (:objects
     	bath_towel.n.01_1 bath_towel.n.01_2 - bath_towel.n.01
    	drawer1 drawer2 drawer3 drawer4 drawer5 drawer6 - drawer
    	jean.n.01_1 jean.n.01_2 jean.n.01_3 jean.n.01_4 - jean.n.01
    	pajama.n.02_1 - pajama.n.02
    	jersey.n.03_1 jersey.n.03_2 jersey.n.03_3 jersey.n.03_4 jersey.n.03_5 jersey.n.03_6 - jersey.n.03
    	sweater1 sweater2 - sweater
    	sock.n.01_1 sock.n.01_2 sock.n.01_3 sock.n.01_4 sock.n.01_5 sock.n.01_6 - sock.n.01
    	pomade1 - pomade
    	watch1 - watch
    	shaver1 - shaver
    	paper_clip.n.01_1 paper_clip.n.01_2 paper_clip.n.01_3 - paper_clip.n.01
    	necktie1 necktie2 - necktie
    	underwear.n.01_1 underwear.n.01_2 underwear.n.01_3 - underwear
    	cabinet.n.01_1 - cabinet.n.01
    	shelf.n.01_1 shelf.n.01_2 - shelf.n.01
    )

    (:init
        (inside bath_towel.n.01_1 drawer2)
        (inside bath_towel.n.01_2 drawer1)
        (inside jean.n.01_1 drawer6)
        (inside jean.n.01_2 drawer5)
        (inside jean.n.01_3 drawer4)
        (inside jean.n.01_4 drawer3)
        (inside pajama.n.02_1 drawer2)
        (inside jersey.n.03_1 drawer1)
        (inside jersey.n.03_2 drawer6)
        (inside jersey.n.03_3 drawer5)
        (inside jersey.n.03_4 drawer4)
        (inside jersey.n.03_5 drawer3)
        (inside jersey.n.03_6 drawer2)
        (inside sweater1 drawer1)
        (inside sweater2 drawer6)
        (inside sock.n.01_1 drawer5)
        (inside sock.n.01_2 drawer4)
        (inside sock.n.01_3 drawer3)
        (inside sock.n.01_4 drawer2)
        (inside sock.n.01_5 drawer1)
        (inside sock.n.01_6 drawer6)
        (inside pomade1 drawer5)
        (inside watch1 drawer4)
        (inside shaver1 drawer3)
        (inside paper_clip.n.01_1 drawer2)
        (inside paper_clip.n.01_2 drawer1)
        (inside paper_clip.n.01_3 drawer6)
        (inside necktie1 drawer5)
        (inside necktie2 drawer4)
        (inside underwear.n.01_1 drawer3)
        (inside underwear.n.01_2 drawer2)
        (inside underwear.n.01_3 drawer6)
        (inside drawer1 cabinet.n.01_1)
        (inside drawer2 cabinet.n.01_1)
        (inside drawer3 cabinet.n.01_1)
        (inside drawer4 cabinet.n.01_1)
        (inside drawer5 cabinet.n.01_1)
        (inside drawer6 cabinet.n.01_1)
        (inside bath_towel.n.01_1 cabinet.n.01_1)
        (inside bath_towel.n.01_2 cabinet.n.01_1)
        (inside jean.n.01_1 cabinet.n.01_1)
        (inside jean.n.01_2 cabinet.n.01_1)
        (inside jean.n.01_3 cabinet.n.01_1)
        (inside jean.n.01_4 cabinet.n.01_1)
        (inside pajama.n.02_1 cabinet.n.01_1)
        (inside jersey.n.03_1 cabinet.n.01_1)
        (inside jersey.n.03_2 cabinet.n.01_1)
        (inside jersey.n.03_3 cabinet.n.01_1)
        (inside jersey.n.03_4 cabinet.n.01_1)
        (inside jersey.n.03_5 cabinet.n.01_1)
        (inside jersey.n.03_6 cabinet.n.01_1)
        (inside sweater1 cabinet.n.01_1)
        (inside sweater2 cabinet.n.01_1)
        (inside sock.n.01_1 cabinet.n.01_1)
        (inside sock.n.01_2 cabinet.n.01_1)
        (inside sock.n.01_3 cabinet.n.01_1)
        (inside sock.n.01_4 cabinet.n.01_1)
        (inside sock.n.01_5 cabinet.n.01_1)
        (inside sock.n.01_6 cabinet.n.01_1)
        (inside pomade1 cabinet.n.01_1)
        (inside watch1 cabinet.n.01_1)
        (inside shaver1 cabinet.n.01_1)
        (inside paper_clip.n.01_1 cabinet.n.01_1)
        (inside paper_clip.n.01_2 cabinet.n.01_1)
        (inside paper_clip.n.01_3 cabinet.n.01_1)
        (inside necktie1 cabinet.n.01_1)
        (inside necktie2 cabinet.n.01_1)
        (inside underwear.n.01_1 cabinet.n.01_1)
        (inside underwear.n.01_2 cabinet.n.01_1)
        (inside underwear.n.01_3 cabinet.n.01_1)
        (inroom cabinet.n.01_1 bedroom)
        (inroom shelf.n.01_1 bedroom)
        (inroom shelf.n.01_2 bedroom)
    )

    (:goal
        (and
            (exists
                (?drawer - drawer)
                (forall
                    (?jean.n.01 - jean.n.01)
                    (inside ?jean.n.01 ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?pajama.n.02 - pajama.n.02)
                    (inside ?pajama.n.02 ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?jersey.n.03 - jersey.n.03)
                    (inside ?jersey.n.03 ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?sweater - sweater)
                    (inside ?sweater ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?sock.n.01 - sock.n.01)
                    (inside ?sock.n.01 ?drawer)
                )
            )
            (ontop ?pomade1 ?shelf.n.01_1)
            (ontop ?watch1 ?shelf.n.01_1)
            (ontop ?shaver1 ?shelf.n.01_1)
            (exists
                (?shelf.n.01 - shelf.n.01)
                (forall
                    (?paper_clip.n.01 - paper_clip.n.01)
                    (ontop ?paper_clip.n.01 ?shelf.n.01)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?necktie - necktie)
                    (inside ?necktie ?drawer)
                )
            )
            (exists
                (?drawer - drawer)
                (forall
                    (?underwear.n.01 - underwear.n.01)
                    (inside ?underwear.n.01 ?drawer)
                )
            )
            (exists
                (?shelf.n.01 - shelf.n.01)
                (forall
                    (?bath_towel.n.01 - bath_towel.n.01)
                    (ontop ?bath_towel.n.01 ?shelf.n.01)
                )
            )
        )
    )
)
