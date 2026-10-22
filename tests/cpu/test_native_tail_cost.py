import copy
import unittest
from src.weave.native_tail_cost import Plan, candidates, dag_predict, select_joint, validate_surface


def curve(plan, down=10., comm=3.):
    return dict(plan=plan.to_dict(), features=[48.,6.,8.,384.,8.], front_us=5.,
                prepare_us=1., planner_us=2., finish_us=1., role_switch_us=0.,
                down_us=[down]*plan.chunks, finalize_us=[1.]*plan.chunks,
                fetch_reduce_us=[comm]*plan.chunks, fetch_result_us=[1.]*plan.chunks)


class CostTests(unittest.TestCase):
    def test_candidate_mechanisms_and_alignment(self):
        plans=candidates(workers=6, hidden=4096, n_tile=128)
        self.assertIn(Plan(0,1,0,True), plans)
        self.assertTrue(all(p.communication_workers<6 for p in plans))
        self.assertFalse(any(p.communication_workers==1 and p.steal_tiles for p in plans))
        with self.assertRaises(ValueError): candidates(6,hidden=192,n_tile=128)

    def test_dag_waits_for_slowest_source_and_gather(self):
        p=Plan(2,1,0)
        rows=[curve(p) for _ in range(4)]
        rows[3]['down_us']=[20.]
        # max(front=5)+planner2+prepare1 + down20+finalize1 + pull/reduce3 + gather1 + finish1
        self.assertEqual(dag_predict(rows),34.)

    def test_chunks_do_not_shrink_expert_groups(self):
        p=Plan(2,2,0)
        rows=[curve(p,down=10.,comm=3.) for _ in range(4)]
        self.assertEqual(dag_predict(rows),35.)
        self.assertEqual(rows[0]['features'][2],8.)

    def test_peer_arrival_wait_is_not_added_twice(self):
        p=Plan(2,1,0)
        rows=[curve(p,down=1.,comm=1.) for _ in range(4)]
        for r in rows:r.update(front_us=1.,planner_us=10.,prepare_us=0.,finish_us=0.)
        rows[3].update(front_us=10.,planner_us=1.)
        # Complete prefix is11, then down1+finalize1+reduce1+gather1.
        self.assertEqual(dag_predict(rows),15.)

    def test_candidate_workloads_need_not_match(self):
        plans=candidates(6)
        surface=[[curve(p),curve(p,down=12.)] for p in plans]
        surface[2][0]['features']=[1.,1.,1.,1.,1.]
        validate_surface(surface,plans,total_rows=192)

    def test_joint_respects_common_chunk_but_rank_specific_roles(self):
        plans=[Plan(0,1,0,True),Plan(1,1,0),Plan(2,1,1),Plan(2,2,0)]
        surfaces=[]
        for r in range(4):
            surfaces.append([[curve(p,down=(2. if i==(1 if r%2 else 2) else 50.))]*2
                              for i,p in enumerate(plans)])
        result=select_joint(surfaces,[[48.,6.,8.,384.,8.]]*4,plans,192)
        self.assertEqual(result['indices'],[2,1,2,1])
        self.assertGreater(result['predicted_us'],0)

    def test_missing_or_fabricated_cost_rejected(self):
        p=Plan(1,1,0)
        with self.assertRaisesRegex(ValueError,'run_native_tail.py'):
            validate_surface([], [p],192)
        bad=curve(p);bad['down_us']=[float('nan')]
        with self.assertRaises(ValueError):validate_surface([[bad,bad]],[p],192)


class HostPlannerTests(unittest.TestCase):
    def test_compiled_planner_matches_python_on_skewed_workloads(self):
        import pathlib, random, shutil, subprocess, tempfile
        from src.weave.native_tail_cost import encode_surface
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('host C++ compiler unavailable')
        root=pathlib.Path(__file__).resolve().parents[2]
        plans=candidates(132)
        randomizer=random.Random(3193)
        with tempfile.TemporaryDirectory() as directory:
            binary=pathlib.Path(directory)/'cost'
            subprocess.run([compiler,'-std=c++17','-O2','-I',str(root),str(root/'tests/cpu/native_tail_cost_host.cpp'),'-o',str(binary)],check=True,capture_output=True,text=True)
            for case in range(24):
                features=[[randomizer.randint(16,64),1.,4.,200.,12.] for _ in range(4)]
                surfaces=[]
                for rank in range(4):
                    surface=[]
                    for p in plans:
                        points=[]
                        for anchor in range(4):
                            row=curve(p,down=randomizer.uniform(.5,20.),comm=randomizer.uniform(.1,4.))
                            row['features']=[float(anchor*16+16),1.,4.,200.,12.]
                            points.append(row)
                        surface.append(points)
                    surfaces.append(surface)
                expected=select_joint(surfaces,features,plans,192)
                values=[len(plans),192]
                for p in plans:values += [p.communication_workers,p.chunks,p.steal_tiles,int(p.serial)]
                values += [4]*(len(plans)*4)
                for f in features:values+=f
                for surface in surfaces:
                    for anchors in encode_surface(surface,plans,192):
                        for row in anchors:values+=row
                result=subprocess.run([str(binary)],input=' '.join(map(str,values)),capture_output=True,text=True,check=True).stdout.split()
                self.assertEqual(list(map(int,result[1:])),expected['indices'],case)
                self.assertAlmostEqual(float(result[0]),expected['predicted_us'],places=11)

    def test_near_tie_keeps_exact_cpu_addition_order(self):
        import pathlib, shutil, subprocess, tempfile
        from src.weave.native_tail_cost import encode_surface
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('host C++ compiler unavailable')
        root=pathlib.Path(__file__).resolve().parents[2]
        plans=[Plan(0,1,0,True),Plan(1,1,0)]
        points=[]
        for i,p in enumerate(plans):
            a=curve(p);a.update(features=[1.]*5,front_us=13.775002159968736,prepare_us=0.,
                planner_us=0.,finish_us=0.,role_switch_us=0.,fetch_reduce_us=[.125],fetch_result_us=[0.])
            a['down_us']=[40.53636780653068 if i==0 else 40.53636780653068+34.56057338663036]
            a['finalize_us']=[34.56057338663036 if i==0 else 0.]
            points.append([a,a])
        surfaces=[points]*4;features=[[1.]*5]*4
        expected=select_joint(surfaces,features,plans,192)
        self.assertEqual(expected['indices'],[1,1,1,1])
        values=[2,192]
        for p in plans:values += [p.communication_workers,p.chunks,p.steal_tiles,int(p.serial)]
        values += [2]*8+[1.]*20
        for surface in surfaces:
            for anchors in encode_surface(surface,plans,192):
                for row in anchors:values+=row
        with tempfile.TemporaryDirectory() as directory:
            binary=pathlib.Path(directory)/'cost'
            subprocess.run([compiler,'-std=c++17','-O2','-ffp-contract=off','-I',str(root),str(root/'tests/cpu/native_tail_cost_host.cpp'),'-o',str(binary)],check=True,capture_output=True)
            result=subprocess.run([str(binary)],input=' '.join(map(str,values)),capture_output=True,text=True,check=True).stdout.split()
            self.assertEqual(list(map(int,result[1:])),expected['indices'])
