package SS_Calculation;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import org.openrdf.model.URI;
import slib.graph.algo.extraction.rvf.instances.InstancesAccessor;
import slib.graph.algo.extraction.rvf.instances.impl.InstanceAccessor_RDF_TYPE;
import slib.graph.algo.utils.GAction;
import slib.graph.algo.utils.GActionType;
import slib.graph.algo.utils.GraphActionExecutor;
import slib.graph.io.conf.GDataConf;
import slib.graph.io.loader.GraphLoaderGeneric;
import slib.graph.io.loader.annot.GraphLoader_TSVannot;
import slib.graph.io.util.GFormat;
import slib.graph.model.graph.G;
import slib.graph.model.impl.graph.memory.GraphMemory;
import slib.graph.model.impl.repo.URIFactoryMemory;
import slib.graph.model.repo.URIFactory;
import slib.sml.sm.core.engine.SM_Engine;
import slib.sml.sm.core.metrics.ic.utils.IC_Conf_Corpus;
import slib.sml.sm.core.metrics.ic.utils.IC_Conf_Topo;
import slib.sml.sm.core.metrics.ic.utils.ICconf;
import slib.sml.sm.core.utils.SMConstants;
import slib.sml.sm.core.utils.SMconf;
import slib.utils.ex.SLIB_Ex_Critic;
import slib.utils.ex.SLIB_Exception;
import slib.utils.impl.Timer;
import java.util.HashMap;



/*
This class is responsible for calculating SSM for a list of protein pairs files of the same species (using same GAF file)
 */
class Calculate_sim_pair {
    // 4 arguments: path_ontology , path_annot , SSM_file and path_dataset.
    // path_ontology is the ontology file path
    // path_annot is the annotations file path in GAF 2.0 version
    // path_dataset is the dataset file path with the pairs of proteins. The format of each line of the dataset files is "Prot1  Prot2   Proxy"
    // SSM_file is the semantic similarity file path for each element of the datasets_files
    public String path_ontology;
    public String path_annot;
    public String namespace;
    public String namespace_uri;
    public String annot_format;
    public String SSM_file;
    public String path_dataset;
    public String SSM;
    public ArrayList<String> aspects;

    public Calculate_sim_pair(String arg1, String arg2 , String arg3, String arg4, String arg5, String arg6, String arg7, String arg8, ArrayList<String> arg9){
        path_ontology = arg1;
        path_annot = arg2;
        namespace = arg3;
        namespace_uri = arg4;
        annot_format = arg5;
        path_dataset = arg6;
        SSM_file = arg7;
        SSM = arg8;
        aspects = arg9;
    }

    public void run() throws SLIB_Exception , FileNotFoundException , IOException{

        Timer t = new Timer();
        t.start();

        URIFactory factory = URIFactoryMemory.getSingleton();
        URI graph_uri = factory.getURI("http://");

        factory.loadNamespacePrefix(namespace, namespace_uri.toString());

        GDataConf ontConf =  new GDataConf(GFormat.RDF_XML , path_ontology);
        GDataConf annotConf;

        if (annot_format.equals("gaf")){
            annotConf = new GDataConf(GFormat.GAF2, path_annot);
        } else if (annot_format.equals("tsv")){
            annotConf = new GDataConf(GFormat.TSV_ANNOT, path_annot);
			annotConf.addParameter(GraphLoader_TSVannot.PARAM_PREFIX_SUBJECT, "http://");
            annotConf.addParameter(GraphLoader_TSVannot.PARAM_PREFIX_OBJECT, namespace_uri);
            annotConf.addParameter(GraphLoader_TSVannot.PARAM_HEADER,"false");
        } else{
            annotConf = new GDataConf(GFormat.GAF2, path_annot);
        }

        ArrayList<G> graphs_aspect =new ArrayList<G>();
        for (String aspect : aspects) {

            G graph =  new GraphMemory(graph_uri);
            GraphLoaderGeneric.populate(ontConf , graph);
            GraphLoaderGeneric.populate(annotConf, graph);

            URI aspectTerm = factory.getURI(aspect);
            GAction reduction_aspect = new GAction(GActionType.VERTICES_REDUCTION);
            reduction_aspect.addParameter("root_uri", aspectTerm.stringValue());
            GraphActionExecutor.applyAction(factory, reduction_aspect, graph);

            graphs_aspect.add(graph);
        }

        if(SSM.equals("all")){
            ArrayList<String> pair_ents = get_ents_dataset(path_dataset);
            semantic_measures_2ents(graphs_aspect, factory, pair_ents, SSM_file);

        }else{
            System.out.println(path_dataset);
            System.out.println(SSM_file);
            ArrayList<String> pair_ents = get_ents_dataset(path_dataset);
            given_semantic_measures_2ents(graphs_aspect, factory, pair_ents, SSM_file, SSM);

        }
        t.stop();
        t.elapsedTime();
    }

    private ArrayList<String> get_ents_dataset(String dataset_filename) throws  IOException{

        FileInputStream file_dataset = new FileInputStream(dataset_filename);
        BufferedReader br = new BufferedReader(new InputStreamReader(file_dataset));

        ArrayList<String> pairs_ents = new ArrayList<>();
        String strLine;

        // Read file line by line
        while ((strLine = br.readLine()) != null) {
            strLine = strLine.substring(0 , strLine.length()-1);
            pairs_ents.add(strLine);
        }
        file_dataset.close();
        return pairs_ents;
    }

    private void groupwise_measure_file(ArrayList<G> graphs_aspect, ArrayList<SM_Engine> engines_aspect, URIFactory factory, ArrayList<String> pairs_ents, String SSM_file, SMconf ssm) throws SLIB_Ex_Critic, FileNotFoundException {

        double sim;
        PrintWriter file = new PrintWriter(SSM_file);

        for (String pair : pairs_ents) {
            ArrayList<String> ents = new ArrayList<String>(Arrays.asList(pair.split("\t")));
            String uri_ent1 = "http://" + ents.get(0);
            String uri_ent2 = "http://" + ents.get(1);

            URI instance1 = factory.getURI(uri_ent1);
            URI instance2 = factory.getURI(uri_ent2);

            file.print(instance1 + "\t" + instance2);
            int i = 0;
            for (G graph: graphs_aspect){
                SM_Engine engine = engines_aspect.get(i);

                InstancesAccessor instance1_acessor = new InstanceAccessor_RDF_TYPE(graph);
                Set<URI> annotations_instance1 = instance1_acessor.getDirectClass(instance1);
                InstancesAccessor instance2_acessor = new InstanceAccessor_RDF_TYPE(graph);
                Set<URI> annotations_instance2 = instance2_acessor.getDirectClass(instance2);

                if (instance1.equals(instance2)) {
                    sim = 1;
                } else{
                    if (annotations_instance1.isEmpty() || annotations_instance2.isEmpty()) {
                        sim = 0;
                    } else {
                        sim = engine.compare(ssm, annotations_instance1, annotations_instance2);
                    }
                }
                file.print("\t" + sim);
                i++;
            }
            file.print("\n");
        }
        file.close();
    }

    private void pairwise_measure_file(ArrayList<G> graphs_aspect, ArrayList<SM_Engine> engines_aspect,  URIFactory factory, ArrayList<String> pairs_ents, String SSM_file, SMconf ssm, SMconf aggregation) throws SLIB_Ex_Critic, FileNotFoundException {

        double sim;
        PrintWriter file = new PrintWriter(SSM_file);

        for (String pair : pairs_ents) {
            ArrayList<String> ents = new ArrayList<String>(Arrays.asList(pair.split("\t")));
            String uri_ent1 = "http://" + ents.get(0);
            String uri_ent2 = "http://" + ents.get(1);

            URI instance1 = factory.getURI(uri_ent1);
            URI instance2 = factory.getURI(uri_ent2);

            file.print(instance1 + "\t" + instance2);
            int i = 0;
            for (G graph : graphs_aspect) {
                SM_Engine engine = engines_aspect.get(i);

                InstancesAccessor instance1_acessor = new InstanceAccessor_RDF_TYPE(graph);
                Set<URI> annotations_instance1 = instance1_acessor.getDirectClass(instance1);
                InstancesAccessor instance2_acessor = new InstanceAccessor_RDF_TYPE(graph);
                Set<URI> annotations_instance2 = instance2_acessor.getDirectClass(instance2);

                if (instance1.equals(instance2)) {
                    sim = 1;
                } else {
                    if (annotations_instance1.isEmpty() || annotations_instance2.isEmpty()) {
                        sim = 0;
                    } else {
                        sim = engine.compare(aggregation, ssm, annotations_instance1, annotations_instance2);
                    }
                }
                file.print("\t" + sim);
                i++;
            }
            file.print("\n");
        }
        file.close();
    }

    private void semantic_measures_2ents(ArrayList<G> graphs_aspect, URIFactory factory, ArrayList<String> pairs_ents, String SSM_file) throws SLIB_Ex_Critic, FileNotFoundException {

        ICconf ic_Seco = new IC_Conf_Topo("Seco", SMConstants.FLAG_ICI_SECO_2004);
        ICconf ic_Resnik = new IC_Conf_Corpus("resnik", SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED);

        SMconf SimGIC_icSeco = new SMconf("gic", SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icSeco.setICconf(ic_Seco);

        SMconf SimGIC_icResnik = new SMconf("gic", SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icResnik.setICconf(ic_Resnik);

        SMconf Resnik_icSeco = new SMconf("resnik", SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icSeco.setICconf(ic_Seco);

        SMconf Resnik_icResnik = new SMconf("resnik", SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icResnik.setICconf(ic_Resnik);

        SMconf max = new SMconf("max", SMConstants.FLAG_SIM_GROUPWISE_MAX);
        SMconf bma = new SMconf("bma", SMConstants.FLAG_SIM_GROUPWISE_BMA);

        ArrayList<SM_Engine> engines_aspect =new ArrayList<SM_Engine>();
        for(G graph: graphs_aspect){
            SM_Engine engine = new SM_Engine(graph);
            engines_aspect.add(engine);
        }

        String filename_ResnikMax_icSeco = SSM_file + "/ss_ResnikMax_ICSeco.txt";
        pairwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_ResnikMax_icSeco, Resnik_icSeco, max);
//        String filename_ResnikMax_icResnik = SSM_file + "/ss_ResnikMax_ICResnik.txt";
//        pairwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_ResnikMax_icResnik, Resnik_icResnik, max);

        String filename_ResnikBMA_icSeco = SSM_file + "/ss_ResnikBMA_ICSeco.txt";
        pairwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_ResnikBMA_icSeco, Resnik_icSeco, bma);
//        String filename_ResnikBMA_icResnik = SSM_file + "/ss_ResnikBMA_ICResnik.txt";
//        pairwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_ResnikBMA_icResnik, Resnik_icResnik, bma);

        String filename_SimGIC_icSeco = SSM_file + "/ss_simGIC_ICSeco.txt";
        groupwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_SimGIC_icSeco, SimGIC_icSeco);
//        String filename_SimGIC_icResnik = SSM_file + "/ss_simGIC_ICResnik.txt";
//        groupwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_SimGIC_icResnik, SimGIC_icResnik);

    }

    private void given_semantic_measures_2ents(ArrayList<G> graphs_aspect, URIFactory factory, ArrayList<String> pairs_ents, String SSM_file, String SSM) throws SLIB_Ex_Critic , FileNotFoundException{

        String[] components_ssm = SSM.split("_");
        ICconf ic;
        SMconf aggregation;
        SMconf ssm;

        HashMap<String, String> flags_ssm = new HashMap<String, String>();
        flags_ssm.put("max", SMConstants.FLAG_SIM_GROUPWISE_MAX);
        flags_ssm.put("bma"  , SMConstants.FLAG_SIM_GROUPWISE_BMA);
        flags_ssm.put("avg"  , SMConstants.FLAG_SIM_GROUPWISE_AVERAGE);
        flags_ssm.put("bmm"  , SMConstants.FLAG_SIM_GROUPWISE_BMM);
        flags_ssm.put("min"  , SMConstants.FLAG_SIM_GROUPWISE_MIN);

        flags_ssm.put("gic" , SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        flags_ssm.put("lee", SMConstants.FLAG_SIM_GROUPWISE_DAG_LEE_2004);
        flags_ssm.put("deane", SMConstants.FLAG_SIM_GROUPWISE_DAG_ALI_DEANE);

        flags_ssm.put("resnik" , SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        flags_ssm.put("li" , SMConstants.FLAG_SIM_PAIRWISE_DAG_EDGE_LI_2003);
        flags_ssm.put("pekarstaab" , SMConstants.FLAG_SIM_PAIRWISE_DAG_EDGE_PEKAR_STAAB_2002);
        flags_ssm.put("rada" , SMConstants.FLAG_SIM_PAIRWISE_DAG_EDGE_RADA_1989);

        flags_ssm.put("ICseco" , SMConstants.FLAG_ICI_SECO_2004);
        flags_ssm.put("ICresnik" , SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED);
        flags_ssm.put("ICharispe", 	SMConstants.FLAG_ICI_HARISPE_2012);
        flags_ssm.put("ICsanchez", SMConstants.FLAG_ICI_SANCHEZ_2011);
        flags_ssm.put("ICzhou", SMConstants.FLAG_ICI_ZHOU_2008);

        ArrayList<SM_Engine> engines_aspect =new ArrayList<SM_Engine>();
        for(G graph: graphs_aspect){
            SM_Engine engine = new SM_Engine(graph);
            engines_aspect.add(engine);
        }

        if (components_ssm.length > 2){
            ic = new IC_Conf_Topo(flags_ssm.get(components_ssm[2]));
            if (ic.isCorpusBased()){
                ic = new IC_Conf_Corpus(flags_ssm.get(components_ssm[2]));
            }
            ssm = new SMconf(flags_ssm.get(components_ssm[0]));
            ssm.setICconf(ic);
            aggregation = new SMconf(flags_ssm.get(components_ssm[1]));

            String filename_SSM = SSM_file + "/ss_" + SSM + ".txt";
            pairwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_SSM, ssm, aggregation);

        }else{
            ic = new IC_Conf_Topo(flags_ssm.get(components_ssm[1]));
            if (ic.isCorpusBased()){
                ic = new IC_Conf_Corpus(flags_ssm.get(components_ssm[1]));
            }
            ssm = new SMconf(flags_ssm.get(components_ssm[0]));
            ssm.setICconf(ic);
            String filename_SSM = SSM_file + "/ss_" + SSM + ".txt";
            groupwise_measure_file(graphs_aspect, engines_aspect, factory, pairs_ents, filename_SSM, ssm);
        }
    }
}



public class Run_SS_calculation_SAs {
    // 5 arguments: args[0], args[1], args[2], args[3], args[4]
    // args[0] - ontology file path
    // arg[1] - annotations file path
    // arg[2] - prefix namespace
    // arg[3]- namespace uri
    // arg[4] - annotations file format ("gaf" or "tsv")
    // arg[5] - dataset file path with the pairs of entities. The format of each line of the dataset files is "Ent1  Ent2   Proxy"
    // arg[6] - new semantic similarity file path
    // arg[7] - SSM (SSM designation or "all")
    // arg[8] - number n of aspects
    // arg[9] - aspect 1
    // arg[10+n] - aspect n


    public static void main(String[] args) throws Exception {

        Calculate_sim_pair datasets;

        int n = Integer.parseInt(args[8]);
        ArrayList<String> aspects =new ArrayList<String>();
        for (int i = 0; i < n; i++) {
            aspects.add(args[9+i]);
        }
		
        datasets = new Calculate_sim_pair(args[0], args[1], args[2],  args[3], args[4], args[5], args[6],args[7], aspects);
        datasets.run();
    }
}
