import argparse,json,concurrent.futures
from pathlib import Path
from .pipeline import inventory,convert_case,write_training_index
from .common import write_json
from .workers import convert_job,initialize_renderer,render_job,track_job


def main():
    p=argparse.ArgumentParser(description='Offline game-state unification and proxy rendering')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['scan','convert','batch']:
        s=sub.add_parser(name);s.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/local.json')
        s.add_argument('--game',choices=['cp2077','gta5','rdr2']);s.add_argument('--limit',type=int)
        if name!='scan':
            s.add_argument('--out',type=Path);s.add_argument('--workers',type=int,default=3)
            s.add_argument('--render-workers',type=int,default=3)
    s=sub.add_parser('render');s.add_argument('--episode',type=Path,required=True);s.add_argument('--view',default='main',choices=['main','monitor','ego','follow'])
    s.add_argument('--width',type=int,default=384);s.add_argument('--height',type=int,default=216);s.add_argument('--limit',type=int)
    s=sub.add_parser('index');s.add_argument('--root',type=Path,required=True)
    s=sub.add_parser('validate');s.add_argument('--root',type=Path,required=True);s.add_argument('--workers',type=int,default=3)
    s=sub.add_parser('tracks');s.add_argument('--root',type=Path,required=True);s.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    if a.command=='render':
        from .render import render_episode
        print(json.dumps(render_episode(a.episode,a.view,a.width,a.height,limit=a.limit)));return
    if a.command=='index':print(json.dumps(write_training_index(a.root)));return
    if a.command=='tracks':
        errors=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(track_job,p.parent):p.parent for p in sorted(a.root.glob('*/*/episode.json'))}
            for f in concurrent.futures.as_completed(futures):
                try:print(json.dumps(f.result()),flush=True)
                except Exception as e:errors.append({'episode':str(futures[f]),'error':repr(e)})
        write_json(a.root/'track_errors.json',errors)
        if errors:raise RuntimeError(str(errors))
        return
    if a.command=='validate':
        from .validate import validate_dataset
        result=validate_dataset(a.root,a.workers)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
        if result['structural_failures']:raise SystemExit(1)
        return
    if not a.config.is_file():p.error('Missing config: copy configs/example.json to configs/local.json and set your dataset roots')
    config=json.loads(a.config.read_text(encoding='utf-8'));items=inventory(config)
    if a.game:items=[x for x in items if x['game']==a.game]
    if a.limit:items=items[:a.limit]
    if a.command=='scan':
        print(json.dumps({'counts':{g:sum(i['game']==g for i in items) for g in config['datasets']},'cases':items},ensure_ascii=False,indent=2));return
    root=(a.out or (a.config.resolve().parent/config['output'])).resolve();root.mkdir(parents=True,exist_ok=True)
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(convert_job,(i['game'],i['source'],root)):i for i in items}
        for f in concurrent.futures.as_completed(futures):
            try:print(json.dumps(f.result(),ensure_ascii=True),flush=True)
            except Exception as e:errors.append({**futures[f],'error':repr(e)});print(json.dumps(errors[-1],ensure_ascii=True),flush=True)
    write_json(root/'conversion_errors.json',errors)
    if a.command=='batch':
        opts=config.get('render',{});width=opts.get('width',384);height=opts.get('height',216)
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.render_workers,initializer=initialize_renderer,initargs=(width,height)) as pool:
            futures={}
            for i in items:
                episode=root/i['game']/i['case_id']
                if not (episode/'episode.json').exists():continue
                for view in opts.get('views',['main']):
                    futures[pool.submit(render_job,(episode,view,width,height))]={**i,'view':view}
            for f in concurrent.futures.as_completed(futures):
                try:print(json.dumps(f.result(),ensure_ascii=True),flush=True)
                except Exception as e:errors.append({**futures[f],'render_error':repr(e)});print(json.dumps(errors[-1],ensure_ascii=True),flush=True)
    write_json(root/'batch_errors.json',errors)
    print(json.dumps(write_training_index(root)))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
