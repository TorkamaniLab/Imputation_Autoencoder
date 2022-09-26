import re
from pathlib import Path

# regexp to parse VMV1 filenames
# HRC.r1-1.EGA.GRCh37.chr22.haplotypes.50108709-50351375.vcf.VMV1
# source, revision, ega, GRCh37, chromosome, start, end

# whole tile fits into vram
fexpr = re.compile(
            r'([^.]*)\.'
          + r'r([^.]*)\.'
          + r'([^.]*)\.'
          + r'([^.]*)\.'
          + r'chr([0-9]*)\.'
          + r'haplotypes\.'
          + r'([0-9]*)-([0-9]*)\.'
          + r'vcf\.VMV1\.gz'
        )

# large tile split into multiple sub-tiles via local-minima
fexpr2 = re.compile(
            r'([^.]*)\.'
          + r'r([^.]*)\.'
          + r'([^.]*)\.'
          + r'([^.]*)\.'
          + r'chr([0-9]*)\.'
          + r'haplotypes\.'
          + r'[0-9]*-[0-9]*\.'
          + r'm[0-9]*_'
          + r'([0-9]*)-([0-9]*)\.'
          + r'VMV1\.gz'
        )

# large tile split, but local minima search finds a sub-tile that is still too large.
# worst-case scenario - have to use brute force, keeping overlap between tiles and
# using fixed length split
fexpr3 = re.compile(
            r'([^.]*)\.'
          + r'r([^.]*)\.'
          + r'([^.]*)\.'
          + r'([^.]*)\.'
          + r'chr([0-9]*)\.'
          + r'haplotypes\.'
          + r'[0-9]*-[0-9]*\.'
          + r'm[0-9]*\.[0-9]*_'
          + r'([0-9]*)-([0-9]*)\.'
          + r'VMV1\.gz'
        )

def test_names():
  names = ["HRC.r1-1.EGA.GRCh37.chr22.haplotypes.45947135-45974835.vcf.VMV1.gz.tbi",
           "HRC.r1-1.EGA.GRCh37.chr22.haplotypes.45970493-46357165.m330.1_45970536-46279523.VMV1.gz",
           "HRC.r1-1.EGA.GRCh37.chr22.haplotypes.50933275-.vcf.VMV1.gz",
           "HRC.r1-1.EGA.GRCh37.chr22.haplotypes.28191154-29457602.m198_28470509-29043563.VMV1.gz"]

  for n in names:
    n2 = f"/abc/def/{n}"
    ans = parse_name(n2)
    print(ans)

def parse_name(path):
  p = str(Path(path).name)
  m = fexpr.match(p)
  if m is None:
    m = fexpr2.match(p)
  if m is None:
    m = fexpr3.match(p)
  if m is None:
    raise ValueError(f"Unable to match {p}")

  source, revision, ega, dataset, chromosome, start, end = m[1], m[2], m[3], m[4], m[5], m[6], m[7]
  return dict(source=source, revision=revision, ega=ega, dataset=dataset,
              chromosome=chromosome, start=start, end=end)

if __name__=="__main__":
  test_names()
