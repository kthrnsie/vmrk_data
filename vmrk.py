#!/usr/bin/python

"""
Process one VMRK file into a set of summary statistics.
"""

import csv
import numpy as np
import os
import sys
import logging
from collections import OrderedDict

# Outliers are defined to fall outside the interval (low, high)
low = 150
high = 3000

# Degrees of freedom adjustment for standard deviations, should generally be 1
ddof = 1

class Code(object):
    """
    A representation of a stimulus/response code
    """
    def __init__(self, side, congruent, correct):
        self.side = side
        self.congruent = congruent
        self.correct = correct

    def __str__(self):
        return "side=%s congruent=%r correct=%r" % (
            self.side, self.congruent, self.correct)

    def fromSRCodes(a, b):
        if a in (1, 2):
            side = "left"
        else:
            side = "right"
        if a in (1, 3):
            congruent = True
        else:
            congruent = False
        if b in (5, 6):
            correct = True
        else:
            correct = False

        return Code(side, congruent, correct)


class Trial(object):
    """
    A representation of a trial.

    A trial is represented by a mark,  stimulus/response code, and a timestamp.
    """
    def __init__(self, mark, srcode, time):
        self.mark = mark
        self.srcode = srcode
        self.time = time

    def __str__(self):
        return "mark=%s srcode=%s time=%d" % (self.mark, self.srcode, self.time)


class Block(object):
    """
    A representation of a block of trials
    """
    def __init__(self):
        self.Mark = []    # Mark label
        self.Code = []    # Stimulus/response codes
        self.Rtim = []    # Response times
        self.Ntri = []    # Number of responses within a trial

    def filter_outliers(self, low, high):
        """
        Remove outlier trials from the block.

        Outlier trials are replace with None.
        """

        for i, rt in enumerate(self.Rtim):
            if rt < low or rt > high:
                self.Code[i] = None
                self.Rtim[i] = None
                self.Ntri[i] = None

    def query(self, side=None, congruent=None, correct=None, lastcorrect=None):
        """
        Return all response times with a given stimulus/response status.

        Any of the query parameters that is None is ignored.
        """
        rtm = []
        marks = []
        for j, (m, c, x) in enumerate(zip(self.Mark, self.Code, self.Rtim)):
            if c is None:
                continue
            if side is not None and c.side != side:
                continue
            if congruent is not None and c.congruent != congruent:
                continue
            if correct is not None and c.correct != correct:
                continue
            if lastcorrect is not None and j > 0:
                if self.Code[j-1] is None or self.Code[j-1].correct != lastcorrect:
                    continue
            rtm.append(x)
            marks.append(m)

        return rtm, marks

    def __str__(self):
        print(self.RT)


# Need to customize this to handle ddof=1 and n=1 (want 0 not NaN in
# this case).
def std(x):
    if len(x) <= 1:
        return 0
    return np.std(x, ddof=ddof)


def process_trial(qu, block):
    """
    Insert summary values obtained from qu into block.

    Parameters
    ----------
    qu :
        Data from a single trial.
    block :
        All data grouped by trial/response type
    """

    # A trial must start with S99 (fixation cross), then have a
    # stimulus and response.  Return early if this is not the case.
    if len(qu) < 3 or qu[0].srcode != 99:
        return

    # ky is the stimulus and response type
    code = Code.fromSRCodes(qu[1].srcode, qu[2].srcode)

    # Response time for first response, multiplication by 2 is a scale
    # conversion.
    rt = 2 * (qu[2].time - qu[1].time)
    block.Mark.append(qu[0].mark)
    block.Code.append(code)
    block.Rtim.append(rt)
    block.Ntri.append(len(qu))


def collapse_blocks(data):
    """
    Collapse a list of blocks into a single block.
    """
    blk = Block()
    for block in data:
        blk.Mark.extend(block.Mark)
        blk.Code.extend(block.Code)
        blk.Rtim.extend(block.Rtim)
        blk.Ntri.extend(block.Ntri)
    return blk


def summarize_vmrk(filename, data):
    """
    Create summary statistics from a VMRK file.

    The VMRK file must be processed with the process_vmrk method
    before running this function.
    """

    cdata = collapse_blocks(data)

    results = OrderedDict()

    results["sid"] = filename.split(".")[0]

    all_trials = [x for x in cdata.Rtim if x is not None]
    correct_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if k is not None and k.correct]
    error_trials = [x for k,x in zip(cdata.Code, cdata.Rtim) if k is not None and not k.correct]

    results["fcn"] = len(correct_trials)
    results["fen"] = len(error_trials)
    results["facc"] = 100 * results["fcn"] / len(all_trials)

    # All trial summaries
    results["frtm"] = np.mean(all_trials)
    results["frtsd"] = std(all_trials)

    # All correct trial summaries
    results["frtmc"] = np.mean(correct_trials)
    results["frtsdc"] = std(correct_trials)

    # All error trial summaries
    results["frtme"] = np.mean(error_trials)
    results["frtsde"] = std(error_trials)

    # Congruent correct trials
    v, _ = cdata.query(correct=True, congruent=True)
    results["fccn"] = len(v)
    results["fcrtmc"] = np.mean(v)
    results["fcrtsdc"] = std(v)

    # Congruent error trials
    v, _ = cdata.query(correct=False, congruent=True)
    results["fcen"] = len(v)
    results["fcrtme"] = np.mean(v)
    results["fcrtsde"] = std(v)

    # Congruent accuracy
    results["fcacc"] = 100 * results["fccn"] / (results["fccn"] + results["fcen"])

    # Incongruent correct trials
    v, _ = cdata.query(correct=True, congruent=False)
    results["ficn"] = len(v)
    results["firtmc"] = np.mean(v)
    results["firtsdc"] = std(v)

    # Incongruent error trials
    v, _ = cdata.query(correct=False, congruent=False)
    results["fien"] = len(v)
    results["firtme"] = np.mean(v)
    results["firtsde"] = std(v)

    # Incongruent accuracy
    results["fiacc"] = 100 * results["ficn"] / (results["ficn"] + results["fien"])

    # Post correct correct trials
    # (don't count first trial of each block)
    u = [b.query(correct=True, lastcorrect=True) for b in data]
    v = [x[0] for x in u]
    results["fpccn"] = sum([max(0, len(x) - 1) for x in v])

    # Post correct error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False, lastcorrect=True) for b in data]
    v = [x[0] for x in u]
    results["fpcen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpcen"] > 0:
        results["fpcertm"] = sum([sum(x[1:]) for x in v]) / results["fpcen"]
    else:
        results["fpcertm"] = 0.

    # Post error correct trials
    # (don't count first trial of each block)
    u = [b.query(correct=True, lastcorrect=False) for b in data]
    v = [x[0] for x in u]
    results["fpecn"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpecn"] > 0:
        results["fpecrtm"] = sum([sum(x[1:]) for x in v]) / results["fpecn"]
    else:
        results["fpecrtm"] = 0.

    # Post error error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False, lastcorrect=False) for b in data]
    v = [x[0] for x in u]
    results["fpeen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpeen"] > 0:
        results["fpeertm"] = sum([sum(x[1:]) for x in v]) / results["fpeen"]
    else:
        results["fpeertm"] = 0.

    # Post error any trials
    # (don't count first trial of each block)
    u = [b.query(lastcorrect=False) for b in data]
    v = [x[0] for x in u]
    results["fpexn"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpexn"] > 0:
        results["fpexrtm"] = sum([sum(x[1:]) for x in v]) / results["fpexn"]
    else:
        results["fpexrtm"] = 0.

    # Post any error trials
    # (don't count first trial of each block)
    u = [b.query(correct=False) for b in data]
    v = [x[0] for x in u]
    results["fpxen"] = sum([max(0, len(x) - 1) for x in v])
    if results["fpxen"] > 0:
        results["fpxertm"] = sum([sum(x[1:]) for x in v]) / results["fpxen"]
    else:
        results["fpxertm"] = 0.

    # Post correct accuracy
    results["faccpc"] = results["fpccn"] / (results["fpccn"] + results["fpcen"])

    # Post error accuracy
    results["faccpe"] = results["fpecn"] / (results["fpecn"] + results["fpeen"])

    # Post error slowing
    results["fpes"] = results["fpeertm"] - results["fpecrtm"]
    results["fpes2"] = results["fpcertm"] - results["fpecrtm"]
    results["fpes3"] = results["fpxertm"] - results["fpexrtm"]

    # Anticipatory responses
    results["fan"] = np.sum(np.asarray(all_trials) < 150)
    results["faen"] = np.sum(np.asarray(error_trials) < 150)

    # Trials with extra responses
    results["fscn"] = sum(np.asarray([x for x in cdata.Ntri if x is not None]) > 3)

    return results


def process_vmrk(filename):
    """
    Process the VMRK format file with name filename.

    Parameters
    ----------
    filename :
        Name of a vmrk format file.

    Returns
    -------
    data : list of Blocks
        data[j] contains all the data for block j.
    """

    fid = open(filename)
    rdr = csv.reader(fid)

    # Keep track of which block we are in
    blocknum = 0
    dblock = 0

    # Assume that we start in practice mode
    mode = "practice"
    n99 = 0
    n144 = False
    qu, data = [], []
    block = Block()

    for line in rdr:

        # Only process "mark" lines
        if len(line) == 0 or not line[0].startswith("Mk"):
            logging.info("Skipping row: %s" % line)
            continue

        # Lines have format Mk###=type, where type=comment, stimulus
        f0 = line[0].split("=")
        mark = f0[0]
        fl = f0[1].lower()
        if fl == "comment":
            continue
        elif fl == "stimulus":
            pass
        else:
            # Not sure what else exists, log it and move on
            logging.info("Skipping row: %s" % line)
            continue

        # Get the type code, e.g. if S16 then n=16
        f1 = line[1].replace(" ", "")
        stimcode = int(f1[1:])

        if mode == "practice":
            if stimcode == 99:
                n99 += 1
            if n99 == 3 and stimcode == 144:
                n144 = True
            if n99 == 3 and n144 and stimcode == 255:
                mode = "experiment"
                continue

        if mode == "practice":
            continue

        qu.append(Trial(mark, stimcode, int(line[2])))

        # Handle end of block markers
        if stimcode in (144, 255):
            if dblock > 0:
                process_trial(qu[0:-1], block)
                qu = [qu[-1]]
                blocknum += 1
                dblock = 0
                block.filter_outliers(low, high)
                data.append(block)
                block = Block()
            continue
        dblock += 1

        if stimcode == 99:
            process_trial(qu[0:-1], block)
            qu = [qu[-1]]

    # Final trial may not have been processed
    process_trial(qu, block)

    return data


if __name__ == "__main__":

    logging.basicConfig(filename="vmrk.log", level=logging.DEBUG)

    # Get all the vmrk files from the current directory.
    print(os.getcwd())
    files = os.listdir()
    files = [f for f in files if f.lower().endswith(".vmrk")]

    out = open("VMRK_Results.csv", "w")

    results = []
    for i, fname in enumerate(files):

        # Process one file
        data = process_vmrk(fname)
        result = summarize_vmrk(fname, data)

        if i == 0:
            # Write header on first iteration only.
            wtr = csv.writer(out, lineterminator='\n')
            header = [k for k in result]
            wtr.writerow(header)

        # Write the results for the current file.
        wtr.writerow([result[k] for k in result])

    out.close()
