layer = iface.activeLayer()
layer.reload()
layer.dataProvider().reloadData()   # re-read the file; refreshes the feature count
layer.updateExtents()
layer.triggerRepaint()
iface.layerTreeView().refreshLayerSymbology(layer.id())  # updates the count shown in the Layers panel
