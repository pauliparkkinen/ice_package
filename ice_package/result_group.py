from system_commons import comm, size, rank

class ResultGroup:
    def __init__(self, invariant_list,  value):
        """
            list invariant_list:  contains Invariants that are used for sorting, the first one used in 
            this group to sort graphs
            float value -  value of the previous invariant that makes the group unique 
        """
        self.value = value
        self.invariant_list = invariant_list
        self.invariant = self.invariant_list[0]
        self.subgroups = []
        self.results = []
        self.bvs = []
        self.wos  = []
   
        
    def __eq__(self, other_group):
        """ Works only for first level invariant, use with caution """
        return other_group != None and other_group.wos != None and self.wos != None and len(self.invariant_list)  == len(other_group.invariant_list) and self.value == other_group.value
    

    def merge(self, other_group):
        self.wos.extend(other_group.wos)
        other_group.wos = None
            
    
    def belongs_to_group(self, value):
        return self.value == value
    
    def try_subgroups(self,  bond_variables,  destination_level=-1):
        value = self.invariant.get_value(bond_variables)
        subgroup_found = False
        if destination_level > 0:
            destination_level -= 1
        elif destination_level == -1:
            pass
        else:
            return self
        
        for subgroup in self.subgroups:
            if subgroup.belongs_to_group(value):
                return subgroup.try_subgroups(bond_variables,  destination_level=destination_level)

        # subgroup was not_found, create new group or return self
        if len(self.invariant_list) == 1:
            return self
        else:
            subgroup = ResultGroup(self.invariant_list[1:],  value)
            self.subgroups.append(subgroup)
            return subgroup.try_subgroups(bond_variables,  destination_level=destination_level)
            
    def get_subgroups_from_level(self,  level):
        result = []
        #print "Current level %i, result count %i" % (level,  len(self.results))
        if level > 0:
            level -= 1
            
            for subgroup in self.subgroups:
                result.extend(subgroup.get_subgroups_from_level(level))
            return result
        else:
            return self.subgroups
        
        
    def add_result(self,  water_orientations, bond_variables=None):
        self.wos.append(water_orientations)
        if bond_variables != None:
            self.bvs.append(bond_variables)

def merge_groups(groups):
    result = []
    if rank == 0:
        for i, group in enumerate(groups):
            merged = False
            for group_2 in result:
                if group_2 == group:
                    group_2.merge(group)
                    merged = True
                    break
            if not merged:
                result.append(group)
    return result
