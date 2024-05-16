import awkward as ak

from python.TH1utils import split_and_flat


def fill2D(h,events,fill_mode="normal",weight=None,**kwargs):
    hist_obj=h.hist_obj

    def add_axes(mask=None):
        add_data=[]
        if "additional_axes" in h.kwargs:
            for idx, ax in enumerate(h.kwargs["additional_axes"]):
                data=events[*ax[0].split("/")][ax[1]]
                if mask is not None:
                    if isinstance(mask, list):
                        for m in mask:
                            data=data[m]
                    else:
                        data=data[mask]
                data=ak.flatten(ak.drop_none(data))
                add_data.append(data)
                hist_obj.axes[idx+2].label=ax[0]+"/"+ax[1]
        return add_data

    if fill_mode=="normal":
        add_data=add_axes()
        data1=split_and_flat(events,h.collection_name,h.var_name)
        data2=split_and_flat(events,h.collection_name2,h.var_name2)
        hist_obj.fill(data1,data2,*add_data,weight=weight)

    elif fill_mode=="rate_pt_vs_score":
        n_ev=len(events)
        freq_x_bx=2760.0*11246/1000
        pt=events[*h.collection_name.split("/")][h.var_name]
        score=events[*h.collection_name2.split("/")][h.var_name2]

        score_cuts=hist_obj.axes[1].edges[:-1]
        score_centers=hist_obj.axes[1].centers


        for score_idx,score_cut in enumerate(score_cuts):
            score_mask=score>score_cut
            maxpt_mask=ak.argmax(pt[score_mask],axis=1,keepdims=True)
            maxpt=ak.flatten(ak.drop_none(pt[score_mask][maxpt_mask]))
            add_data=add_axes(mask=[score_mask, maxpt_mask])
            for pt_thr,pt_bin_center in zip(hist_obj.axes[0].edges, hist_obj.axes[0].centers):
                hist_obj.fill(pt_bin_center,score_centers[score_idx],*add_data, weight=ak.sum(maxpt>=pt_thr))

        hist_obj.axes[0].label="Online pT cut"
        hist_obj.axes[1].label="Score cut"
        h.name=h.name.rsplit("/",2)[0]+"/rate_pt_vs_score"
        hist_obj=hist_obj*freq_x_bx/n_ev


    return hist_obj
