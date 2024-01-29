import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.systems.system_base import get_system
from scipy.spatial.transform import Rotation as R
import omnigibson.lazy as lazy
import trimesh

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

OBJECTS = [
  'ashtray-dhkkfo',
  'ashtray-nfuxzd',
  'baking_powder_jar-lgopij',
  'baking_sheet-yhurut',
  'barbecue_sauce_bottle-gfxrnj',
  'basil_jar-swytaw',
  'beaker-dtjmai',
  'beaker-effbnc',
  'beaker-exzsal',
  'beaker-fxrsyi',
  'beaker-fyrkzs',
  'beaker-jdijek',
  'beaker-qwoqqr',
  'beaker-rhohgs',
  'beaker-sfvswx',
  'beaker-sstojv',
  'beaker-uobdoq',
  'beaker-uzgibd',
  'beaker-zycgen',
  'beef_broth_carton-ecqxgd',
  'beer_bottle-nigfha',
  'beer_glass-lymciz',
  'beer_glass-mxsliu',
  'beer_glass-rxscji',
  'beer_keg-xtqbuf',
  'bird_feed_bag-dpxnlc',
  'bird_feeder-hvlkjx',
  'blender-cwkvib',
  'blender-dhfqid',
  'blender-eyedvd',
  'blender-xnjqix',
  'bowl-adciys',
  'bowl-ajzltc',
  'bowl-aspeds',
  'bowl-belcml',
  'bowl-bexgtn',
  'bowl-bnobdx',
  'bowl-byzaxy',
  'bowl-ckxwea',
  'bowl-cypjlv',
  'bowl-dalyim',
  'bowl-eawgwj',
  'bowl-eipwho',
  'bowl-fedafr',
  'bowl-feuaak',
  'bowl-fiarri',
  'bowl-fwdfeg',
  'bowl-hitnkv',
  'bowl-hpqjug',
  'bowl-hynhgz',
  'bowl-jblalf',
  'bowl-jfvjep',
  'bowl-jhtxxh',
  'bowl-jpvcjv',
  'bowl-kasebx',
  'bowl-kdkrov',
  'bowl-kthvrl',
  'bowl-lgaxzt',
  'bowl-mspdar',
  'bowl-nkkhbn',
  'bowl-npuuir',
  'bowl-oyidja',
  'bowl-pihjqa',
  'bowl-qzodht',
  'bowl-rbnyxi',
  'bowl-rlwpcd',
  'bowl-sqqahm',
  'bowl-szgdpc',
  'bowl-tvtive',
  'bowl-tyczoo',
  'bowl-vccsrl',
  'bowl-wryghu',
  'bowl-wtepsx',
  'bowl-xplzbo',
  'bowl-xpnlup',
  'brown_rice_sack-pbvpua',
  'brown_sugar_sack-uftzyo',
  'bucket-bdhvnt',
  'bucket-lsmlzi',
  'bucket-wlilma',
  'butter_package-qixpto',
  'can-gqwnfv',
  'can-xcppkc',
  'canteen-ouhqnw',
  'canteen-ttxunv',
  'carafe-hdbsog',
  'carafe-mdtkkv',
  'carafe-ocjcgp',
  'carton-causya',
  'carton-cdmmwy',
  'carton-hhlmbi',
  'carton-libote',
  'carton-msfzpz',
  'carton-sxlklf',
  'carton-uartvl',
  'carton-ylrxhe',
  'casserole-mmbavt',
  'casserole-ncbsee',
  'cast-heuzgu',
  'cat_food_tin-rclizj',
  'catsup_bottle-qfvqfm',
  'cauldron-lbcxwi',
  'cauldron-zndohl',
  'chalice-sfkezf',
  'chia_seed_bag-xkixrg',
  'chicken_broth_carton-ztripg',
  'chicken_soup_carton-ooyqcr',
  'chickpea_can-jeqtzg',
  'china-cvdbum',
  'china-gklybu',
  'china-hacehh',
  'china-jwxbpa',
  'china-qhnpmc',
  'china-qlxhhh',
  'chlorine_bottle-uzkxtz',
  'chocolate_sauce_bottle-yegrkf',
  'cleansing_bottle-nsxhvs',
  'cleansing_bottle-ovjhuf',
  'clove_jar-cqdioi',
  'cocktail_glass-xevdnl',
  'cocoa_box-kxwgoo',
  'cocoa_powder_box-cjmtvq',
  'coconut_oil_jar-phimqa',
  'coffee_bean_jar-loduxu',
  'coffee_cup-ckkwmj',
  'coffee_cup-dkxddg',
  'coffee_cup-fgizgn',
  'coffee_cup-ibhhfj',
  'coffee_cup-nbhcgu',
  'coffee_cup-nhzrei',
  'coffee_cup-rixzrk',
  'coffee_cup-rypdvd',
  'coffee_cup-siksnl',
  'coffee_cup-skamgp',
  'coffee_cup-xjdyon',
  'coffee_cup-ykuftq',
  'cola_bottle-oyqdtz',
  'compost_bin-fvkdos',
  'cooking_oil_bottle-cfdond',
  'copper_pot-gqemcq',
  'cornstarch_jar-dhseui',
  'cream_carton-lfjmos',
  'cream_cheese_box-hfclfn',
  'crock_pot-lspxjq',
  'crock_pot-xdahvv',
  'decanter-ofasfw',
  'detergent_bottle-yufawg',
  'disinfectant_bottle-ifqdxn',
  'dog_food_can-vxxcvg',
  'drip_pot-xmxvml',
  'electric_cauldron-qmdgct',
  'electric_kettle-hkdsla',
  'electric_mixer-ceaeqf',
  'electric_mixer-qornxa',
  'erlenmeyer_flask-bpwjxr',
  'erlenmeyer_flask-iwwpsf',
  'fabric_softener_bottle-uyixwc',
  'feta_box-qxnzpx',
  'floor_wax_bottle-fzhcdb',
  'flour_sack-zhsjcs',
  'food_processor-gamkbo',
  'fragrance_decanter-ngcvaw',
  'frosting_jar-ztyxyi',
  'frying_pan-aewpzn',
  'frying_pan-cprjvq',
  'frying_pan-hzspwg',
  'frying_pan-jpzusm',
  'frying_pan-mhndon',
  'frying_pan-sfbdjn',
  'frying_pan-snvhlz',
  'frying_pan-vycozd',
  'fuel_can-bfaqfe',
  'gelatin_box-oxknkz',
  'glaze_bottle-zdxagk',
  'goblet-nawrfs',
  'graduated_cylinder-egpkea',
  'granola_box-yzeuqo',
  'granulated_sugar_jar-qwthua',
  'granulated_sugar_sack-oywwzz',
  'grated_cheese_sack-fxnjfr',
  'gravy_boat-arryyl',
  'gravy_boat-krgqwl',
  'ground_beef_package-lwsgzd',
  'herbicide_bottle-aeslmf',
  'honey_jar-dhgtvg',
  'hot_sauce_bottle-qvpthd',
  'hot_tub-nuzkuf',
  'hot_tub-wbnkfk',
  'hummus_box-jvnqly',
  'hydrogen_peroxide_bottle-luhnej',
  'ice_bucket-vlurir',
  'ice_cream_carton-lulzdz',
  'ink_bottle-gcyvrx',
  'instant_coffee_jar-ycgxwb',
  'instant_pot-lkbvad',
  'instant_pot-wengzf',
  'jar-acsllv',
  'jar-bnrvcs',
  'jar-bpxhso',
  'jar-bqpmsv',
  'jar-busiti',
  'jar-crlhmi',
  'jar-dlvall',
  'jar-gejwoi',
  'jar-gkakwk',
  'jar-gqtsam',
  'jar-hjrnct',
  'jar-ifgcmr',
  'jar-iuydyz',
  'jar-jdwvyt',
  'jar-jnjtrl',
  'jar-kfzxah',
  'jar-kijnrj',
  'jar-lvuvbf',
  'jar-mefezc',
  'jar-miivhi',
  'jar-mlnuza',
  'jar-mxhrcl',
  'jar-ociqav',
  'jar-pjaljg',
  'jar-qdnmwg',
  'jar-sjwgfn',
  'jar-vxqpnm',
  'jar-vyfehw',
  'jar-waousd',
  'jar-wcqjew',
  'jar-zdeyzf',
  'jelly_bean_jar-nftsal',
  'jelly_jar-lrjoro',
  'jigger-aysfhf',
  'jimmies_jar-oqyoos',
  'jug-gjgwvi',
  'jug-hjjeeh',
  'jug-llexze',
  'jug-pvxfot',
  'jug-quzmfw',
  'kettle-bzisss',
  'kettle-vjbldp',
  'lemon_juice_bottle-qsdqik',
  'lemonade_bottle-yprkek',
  'lime_juice_bottle-bnekjp',
  'liquid_carton-rsvypp',
  'liquid_carton-ykfkyq',
  'liquid_soap_bottle-hazvbh',
  'litter_box-rwnakn',
  'lubricant_bottle-omknho',
  'lunch_box-adxzhe',
  'maple_syrup_jar-wigtue',
  'margarine_box-owqbsb',
  'marinara_jar-cydfkt',
  'measuring_cup-ahtzhp',
  'milk_carton-icvmix',
  'mixing_bowl-bsgybx',
  'mixing_bowl-deudkt',
  'mixing_bowl-xifive',
  'mop_bucket-xjzyfc',
  'mug-dhnxww',
  'mug-ehnmxj',
  'mug-fapsrj',
  'mug-jgethp',
  'mug-kewbyf',
  'mug-kitxam',
  'mug-lgxhsc',
  'mug-ntgftr',
  'mug-ppdqbj',
  'mug-ppzttc',
  'mug-waqrdy',
  'mug-yiamah',
  'mug-yxaapv',
  'mulch_bag-zsrpiu',
  'mustard_bottle-lgxfyv',
  'noodle_jar-tmjxno',
  'oat_box-xdhysb',
  'oden_cooker-fjpams',
  'oil_bottle-gcixra',
  'oil_bottle-yoxfyu',
  'olive_oil_bottle-zfvhus',
  'orange_juice_carton-jjlfla',
  'paper_bag-bzsxgw',
  'paper_bag-ruryqd',
  'paper_bag-wvhmww',
  'paper_cup-guobeq',
  'pasta_box-jaypjo',
  'peanut_butter_jar-xdxqxj',
  'pellet_food_bag-jgyqpd',
  'petfood_bag-fhfqys',
  'petri_dish-vbiqcq',
  'petri_dish-xfqatj',
  'pill_bottle-csvdbe',
  'pill_bottle-wsasmm',
  'pineapple_juice_carton-barzwx',
  'pitcher-ankfvi',
  'pitcher-bbewjo',
  'pitcher-mbrlge',
  'pitcher-ompiss',
  'pitcher-tsyims',
  'pitcher-tzbnmh',
  'pitcher-wmkwhg',
  'plant_pot-ihnfbi',
  'plant_pot-skbcqq',
  'plant_pot-vhglly',
  'plant_pot-ygrtaz',
  'plate-aewthq',
  'plate-akfjxx',
  'plate-amhlqh',
  'plate-aynjhg',
  'plate-bgxzec',
  'plate-dbprwc',
  'plate-dnqekb',
  'plate-efkgcw',
  'plate-eixyyn',
  'plate-eozsdg',
  'plate-fhdyrj',
  'plate-fkpaie',
  'plate-haewxp',
  'plate-iawoof',
  'plate-ihrjrb',
  'plate-itoeew',
  'plate-ivbrtz',
  'plate-ivuveo',
  'plate-iwfvwf',
  'plate-kkjiko',
  'plate-kkmkbd',
  'plate-ksgizx',
  'plate-lixwwc',
  'plate-lkomhp',
  'plate-luhkiz',
  'plate-molqhs',
  'plate-mtetqm',
  'plate-nbuspz',
  'plate-nhodax',
  'plate-nikfgd',
  'plate-nmhxfz',
  'plate-nrjump',
  'plate-ntedfx',
  'plate-odmjdd',
  'plate-pjinwe',
  'plate-pkkgzc',
  'plate-pyilfa',
  'plate-qbxfmv',
  'plate-qtfzeq',
  'plate-qyuyjr',
  'plate-spppps',
  'plate-tgrsui',
  'plate-uakqei',
  'plate-ujodgo',
  'plate-uumkbl',
  'plate-vitdwc',
  'plate-vjqzwa',
  'plate-vtjwof',
  'plate-wgcgia',
  'plate-wqgndf',
  'plate-xfjmld',
  'plate-xtdcau',
  'plate-ypdfrp',
  'plate-zpddxu',
  'platter-csanbr',
  'platter-ekjpdj',
  'platter-hnlivs',
  'platter-iadlti',
  'platter-ieoasd',
  'platter-kiiamx',
  'polish_bottle-hldhxl',
  'popcorn_bag-hdcpqg',
  'pressure_cooker-otyngn',
  'pumpkin_seed_bag-wyojnz',
  'punching_bag-svkdji',
  'raisin_box-yowyst',
  'reagent_bottle-tnjpsf',
  'reagent_bottle-trtrsl',
  'reagent_bottle-uaijua',
  'reagent_bottle-ukayce',
  'reagent_bottle-xstykf',
  'recycling_bin-duugbb',
  'recycling_bin-nuoypc',
  'refried_beans_can-dafdgk',
  'round_bottom_flask-fjytro',
  'round_bottom_flask-hmzafz',
  'round_bottom_flask-injdmj',
  'round_bottom_flask-tqyiso',
  'rum_bottle-ueagnt',
  'saddle_soap_bottle-ugqdao',
  'salad_bowl-dhdhul',
  'salsa_bottle-kydilb',
  'salt_bottle-wdpcmk',
  'saucepan-fsinsu',
  'saucepot-chjetk',
  'saucepot-fbfmwt',
  'saucepot-kvgaar',
  'saucepot-obuxbe',
  'saucepot-ozrwwk',
  'saucepot-pkfydm',
  'saucepot-sthkfz',
  'saucepot-tfzijn',
  'saucepot-urqzec',
  'saucepot-uvzmss',
  'saucepot-vqtkwq',
  'saucepot-wfryvm',
  'saucer-cjsbft',
  'saucer-mgbeah',
  'saucer-oxivmf',
  'saucer-szzjzd',
  'saucer-vghfkh',
  'sesame_oil_bottle-bupgpj',
  'shampoo_bottle-tukaoq',
  'shopping_basket-nedrsh',
  'shopping_basket-vsxhsv',
  'shortening_carton-gswpdr',
  'soap_bottle-hamffy',
  'soda_can-xbkwbi',
  'soda_cup-fsfsas',
  'soda_cup-gnzegv',
  'soda_cup-lpanoc',
  'soda_cup-vicaqs',
  'soda_water_bottle-upfssc',
  'sodium_carbonate_jar-vxtjjn',
  'soil_bag-gzcqwx',
  'solvent_bottle-nbctrk',
  'soy_sauce_bottle-saujjl',
  'specimen_bottle-vlplhs',
  'stockpot-azoiaq',
  'stockpot-dcleem',
  'stockpot-fuzmdd',
  'stockpot-grrcna',
  'stockpot-gxiqbw',
  'stockpot-lfnbhc',
  'stockpot-oshwps',
  'stockpot-yvhmex',
  'sugar_sack-xixblr',
  'sugar_syrup_bottle-kdlbbq',
  'sunflower_seed_bag-dhwlaw',
  'swimming_pool-kohria',
  'swimming_pool-qjhauf',
  'swimming_pool-sbvksi',
  'swimming_pool-vnvmkx',
  'tank-bsdexp',
  'teacup-cpozxi',
  'teacup-kccqwj',
  'teacup-oxfzfe',
  'teacup-tfzfam',
  'teacup-vckahe',
  'teacup-wopjex',
  'teacup-zdvgol',
  'teapot-foaehs',
  'teapot-jlalfc',
  'teapot-mvrhya',
  'test_tube-apybok',
  'test_tube-iejmzf',
  'test_tube-qwtyqj',
  'test_tube-tgodzn',
  'test_tube-vnmcfg',
  'test_tube-ykvekt',
  'toilet_soap_bottle-iyrrna',
  'tomato_paste_can-krarex',
  'tomato_sauce_jar-krfzqk',
  'trash_can-aefcem',
  'trash_can-cdzyew',
  'trash_can-cjmezk',
  'trash_can-djgllo',
  'trash_can-dnvpag',
  'trash_can-eahqyq',
  'trash_can-fkosow',
  'trash_can-gilsji',
  'trash_can-glzckq',
  'trash_can-gsgutn',
  'trash_can-gvnfgj',
  'trash_can-gxajos',
  'trash_can-hqdnjz',
  'trash_can-hxsyxo',
  'trash_can-ifzxzj',
  'trash_can-jlawet',
  'trash_can-leazin',
  'trash_can-mcukuh',
  'trash_can-mdojox',
  'trash_can-pdmzhv',
  'trash_can-rbqckd',
  'trash_can-rteihy',
  'trash_can-uknjdm',
  'trash_can-vasiit',
  'trash_can-wklill',
  'trash_can-wkxtxh',
  'trash_can-xkqkbf',
  'trash_can-zotrbg',
  'tray-avotsj',
  'tray-coqeme',
  'tray-glwebh',
  'tray-gsxbym',
  'tray-hbjdlb',
  'tray-hjxczh',
  'tray-huwhjg',
  'tray-hvlfig',
  'tray-iaaiyi',
  'tray-incirm',
  'tray-jpcflq',
  'tray-mhhoga',
  'tray-mkdcha',
  'tray-mmegts',
  'tray-spopfj',
  'tray-thkphg',
  'tray-tkgsho',
  'tray-txcjux',
  'tray-uekqey',
  'tray-vxbtax',
  'tray-wbwmcs',
  'tray-xzcnjq',
  'tray-yqtlhy',
  'tray-zcmnji',
  'tray-zsddtq',
  'tupperware-mkstwr',
  'vanilla_bottle-drevku',
  'vase-aegxpb',
  'vase-atgnsc',
  'vase-bbduix',
  'vase-bedkqu',
  'vase-cvyops',
  'vase-dfjcsi',
  'vase-dwspgo',
  'vase-dxnzuk',
  'vase-eqhgiy',
  'vase-euqzpy',
  'vase-gopbrh',
  'vase-hkwtnf',
  'vase-hliauj',
  'vase-htyvuz',
  'vase-icpews',
  'vase-ipbgrw',
  'vase-jdddsr',
  'vase-jpwsrp',
  'vase-kjeudr',
  'vase-mawxva',
  'vase-mdmwcs',
  'vase-meetii',
  'vase-nodcpg',
  'vase-nuqzjs',
  'vase-pqsamn',
  'vase-qebiei',
  'vase-rfegnv',
  'vase-rfigof',
  'vase-rusmlm',
  'vase-rwotxo',
  'vase-saenda',
  'vase-sakwru',
  'vase-stqkvx',
  'vase-szsudo',
  'vase-tjrbxv',
  'vase-toreid',
  'vase-twknia',
  'vase-uuypot',
  'vase-vmbzmm',
  'vase-wltgjn',
  'vase-wmuysk',
  'vase-xfduug',
  'vase-ysdoep',
  'vase-zaziny',
  'vase-zwekzu',
  'vinegar_bottle-hbsbwt',
  'vinegar_bottle-ykysuc',
  'vodka_bottle-bojwlu',
  'wading_pool-xixlzr',
  'waffle_maker-yjmnej',
  'washer-dobgmu',
  'washer-jgyzhv',
  'washer-mrgspe',
  'washer-omeuop',
  'washer-xusefg',
  'washer-ynwamu',
  'washer-zgzvcv',
  'washer-ziomqg',
  'water_bottle-ackxiy',
  'water_bottle-lzdzkk',
  'water_glass-bbpraa',
  'water_glass-cdteyb',
  'water_glass-edfzlt',
  'water_glass-elwfms',
  'water_glass-evaida',
  'water_glass-ewgotr',
  'water_glass-ggpnlr',
  'water_glass-gypzlg',
  'water_glass-igyuko',
  'water_glass-imsnkt',
  'water_glass-kttdbu',
  'water_glass-ktuvuo',
  'water_glass-kuiiai',
  'water_glass-lvqgvn',
  'water_glass-nfoydb',
  'water_glass-onbiqg',
  'water_glass-ptciim',
  'water_glass-qbejli',
  'water_glass-slscza',
  'water_glass-szjfpb',
  'water_glass-uwtdng',
  'water_glass-vcwsbm',
  'water_glass-wvztiw',
  'water_glass-ybhepe',
  'water_glass-zbridw',
  'wheelbarrow-msaevo',
  'whiskey_bottle-jpduev',
  'white_rice_sack-xiwkwz',
  'white_sauce_bottle-gtwngf',
  'wine_bottle-hlzfxw',
  'wine_bottle-vjdkci',
  'wine_bottle-zuctnl',
  'wine_sauce_bottle-vqtevv',
  'wineglass-aakcyj',
  'wineglass-adiwil',
  'wineglass-akusda',
  'wineglass-bnored',
  'wineglass-bovcqx',
  'wineglass-cmdagy',
  'wineglass-euzudc',
  'wineglass-exasdr',
  'wineglass-ezsdil',
  'wineglass-ggbdlq',
  'wineglass-hxccge',
  'wineglass-jzmrdd',
  'wineglass-kxovsj',
  'wineglass-oadvet',
  'wineglass-ovoceo',
  'wineglass-vxmzmq',
  'wineglass-yfzibn',
  'wok-pobfpe',
  'yeast_jar-vmajcm',
  'yogurt_carton-ahbhsd'
]

def generate_box(box_half_extent):
    # The floor plane already exists
    # We just need to generate the side planes
    plane_centers = np.array([
        [1, 0, 1],
        [0, 1, 1],
        [-1, 0, 1],
        [0, -1, 1],
    ]) * box_half_extent
    for i, pc in enumerate(plane_centers):
        plane = lazy.omni.isaac.core.objects.ground_plane.GroundPlane(
            prim_path=f"/World/plane_{i}",
            name=f"plane_{i}",
            z_position=0,
            size=box_half_extent[2],
            color=None,
            visible=False,

            # TODO: update with new PhysicsMaterial API
            # static_friction=static_friction,
            # dynamic_friction=dynamic_friction,
            # restitution=restitution,
        )

        plane_as_prim = XFormPrim(
            prim_path=plane.prim_path,
            name=plane.name,
        )
        
        # Build the plane orientation from the plane normal
        horiz_dir = pc - np.array([0, 0, box_half_extent[2]])
        plane_z = -1 * horiz_dir / np.linalg.norm(horiz_dir)
        plane_x = np.array([0, 0, 1])
        plane_y = np.cross(plane_z, plane_x)
        plane_mat = np.array([plane_x, plane_y, plane_z]).T
        plane_quat = R.from_matrix(plane_mat).as_quat()
        plane_as_prim.set_position_orientation(pc, plane_quat)

def generate_particles_in_box(box_half_extent):
    water = get_system("water")
    particle_radius = water.particle_radius

    # Grab the link's AABB (or fallback to obj AABB if link does not have a valid AABB),
    # and generate a grid of points based on the sampling distance
    low = np.array([-1, -1, 0]) * box_half_extent
    high = np.array([1, 1, 2]) * box_half_extent
    extent = np.ones(3) * box_half_extent * 2
    # We sample the range of each extent minus
    sampling_distance = 2 * particle_radius
    n_particles_per_axis = (extent / sampling_distance).astype(int)
    assert np.all(n_particles_per_axis), f"box is too small to sample any particle of radius {particle_radius}."

    # 1e-10 is added because the extent might be an exact multiple of particle radius
    arrs = [np.arange(l + particle_radius, h - particle_radius + 1e-10, particle_radius * 2)
            for l, h, n in zip(low, high, n_particles_per_axis)]
    # Generate 3D-rectangular grid of points
    particle_positions = np.stack([arr.flatten() for arr in np.meshgrid(*arrs)]).T

    water.generate_particles(
        positions=particle_positions,
    )

    return water

def process_object(cat, mdl):
    cfg = {
        "scene": {
            "type": "Scene",
        }
    }

    env = og.Environment(configs=cfg)

    # First import the fillable
    fillable = DatasetObject("fillable", category=cat, model=mdl, kinematic_only=True)
    og.sim.import_object(fillable)
    og.sim.step()

    # Now move it into position
    aabb_extent = fillable.aabb_extent
    obj_bbox_height = aabb_extent[2]
    obj_bbox_center = fillable.aabb_center
    obj_bbox_bottom = obj_bbox_center - np.array([0, 0, obj_bbox_height / 2])
    obj_current_pos = fillable.get_position()
    obj_pos_wrt_bbox = obj_current_pos - obj_bbox_bottom
    obj_dipped_pos = obj_pos_wrt_bbox
    obj_free_pos = obj_pos_wrt_bbox + np.array([0, 0, 2 * aabb_extent[2] + 0.2])
    fillable.set_position_orientation(obj_free_pos, np.array([0, 0, 0, 1]))
    og.sim.step()

    # Now generate the box and the particles
    box_half_extent = aabb_extent * 1.5  # 1.5 times the space
    box_half_extent[2] = np.maximum(box_half_extent[2], 0.1)  # at least 10cm water
    generate_box(box_half_extent)
    og.sim.step()
    water = generate_particles_in_box(box_half_extent)
    for _ in range(100):
        og.sim.step()

    # Move the object down into the box slowly
    lin_vel = 0.02
    while True:
        delta_z = -lin_vel * og.sim.get_rendering_dt()
        cur_pos = fillable.get_position()
        new_pos = cur_pos + np.array([0, 0, delta_z])
        fillable.set_position(new_pos)
        og.sim.step()
        if fillable.get_position()[2] < obj_dipped_pos[2]:
            break

    # Let the particles settle
    for _ in range(100):
        og.sim.step()

    # Now move the object out of the water
    while True:
        delta_z = lin_vel * og.sim.get_rendering_dt()
        cur_pos = fillable.get_position()
        new_pos = cur_pos + np.array([0, 0, delta_z])
        fillable.set_position(new_pos)
        og.sim.step()
        if fillable.get_position()[2] > obj_free_pos[2]:
            break

    # Gentle side-by-side shakeoff
    directions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ])
    for d in directions:
        for _ in range(60):
            delta_pos = lin_vel * og.sim.get_rendering_dt() * d
            cur_pos = fillable.get_position()
            new_pos = cur_pos + delta_pos
            fillable.set_position(new_pos)
            og.sim.step()

    # Let the particles settle
    for _ in range(30):
        og.sim.step()

    # Get the particles whose center is within the object's AABB
    aabb_min, aabb_max = fillable.aabb
    particles = water.get_particles_position_orientation()[0]
    particles = particles[np.all(particles <= aabb_max, axis=1)]
    particles = particles[np.all(particles >= aabb_min, axis=1)]
    assert len(particles) > 0, "No particles found in the AABB of the object."

    # Get the particles in the frame of the object
    particles -= fillable.get_position()

    # Get the convex hull of the particles
    hull = trimesh.convex.convex_hull(particles)

    from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand
    container_prim_path = fillable.root_link.prim_path + "/container"
    CreateMeshPrimWithDefaultXformCommand(prim_path=container_prim_path, prim_type="Mesh", trimesh_mesh=hull).do()
    mesh_prim = XFormPrim(name="container", prim_path=container_prim_path)

    # Now wait for observation
    while True:
        og.sim.render()

    og.sim.clear()

# Create an environment and fill it with balls
def main():
    for obj_name in OBJECTS:
        cat, mdl = obj_name.split("-")
        print(f"Processing {obj_name}")
        process_object(cat, mdl)


if __name__ == "__main__":
    main()