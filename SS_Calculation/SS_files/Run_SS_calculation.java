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

/*
This class is responsible for convertng GAF 2.1 to GAF 2.0, the format accepted by SML
 */
class Convert_GAF_versions {

    // 2 arguments: annot and new_annot
    // annot is the file annotations path in GAF.2.1 version
    // new_annot is the new file annotations path in GAF 2.0 version
    public String annot;
    public String new_annot;

    public Convert_GAF_versions(String arg1, String arg2){
        annot = arg1;
        new_annot = arg2;
    }


    public void run() throws FileNotFoundException , IOException {

        PrintWriter new_file = new PrintWriter(new_annot);
        new_file.println("!gaf-version: 2.0");

        // Open the file with annotations
        FileInputStream file_annot = new FileInputStream(annot);
        BufferedReader br = new BufferedReader(new InputStreamReader(file_annot));

        String strLine;


        // Read file line by line
        while ((strLine = br.readLine()) != null){

            if (!strLine.startsWith("!") || !strLine.isEmpty() || strLine != null  || strLine!= ""){

                ArrayList<String> fields = new ArrayList<String>(Arrays.asList(strLine.split("\t")));

                if (fields.size()>12){ // verify if the annotation have taxon
                    fields.set(7 , fields.get(7).replace("," , "|"));
                    String newLine = String.join("\t" , fields);
                    new_file.println(newLine);
                }
            }
        }

        new_file.close();
        file_annot.close();


    }

}

/*
This class is responsible for calculating SSM for a list of protein pairs files of the same species (using same GAF file)
 */
class Calculate_sim_prot {

    // 4 arguments: path_file_goOBO , annot , SSM_files and dataset_files.
    // path_file_goOBO is the ontology file path
    // annot is the annotations file path in GAF 2.0 version
    // datasets_files is the list of dataset files path with the pairs of proteins. The format of each line of the dataset files is "Prot1  Prot2   Label"
    // SSM_files is the list of semantic similarity files paths for each element of the dasets_files
    public String path_file_goOBO;
    public String annot;
    public String[] SSM_files;
    public String[] dataset_files;

    public Calculate_sim_prot(String arg1, String arg2 , String[] arg3, String[] arg4){
        path_file_goOBO = arg1;
        annot = arg2;
        SSM_files = arg3;
        dataset_files = arg4;
    }

    public void run() throws SLIB_Exception , FileNotFoundException , IOException{

        Timer t = new Timer();
        t.start();

        URIFactory factory = URIFactoryMemory.getSingleton();
        URI graph_uri = factory.getURI("http://go/");

        // define a prefix in order to build valid uris from ids such as GO:XXXXX
        // (the URI associated to GO:XXXXX will be http://go/XXXXX)
        factory.loadNamespacePrefix("GO" , graph_uri.toString());

        G graph_BP = new GraphMemory(graph_uri);
        G graph_CC = new GraphMemory(graph_uri);
        G graph_MF = new GraphMemory(graph_uri);

        GDataConf goConf =  new GDataConf(GFormat.OBO , path_file_goOBO);
        GDataConf annotConf = new GDataConf(GFormat.GAF2, annot);

        GraphLoaderGeneric.populate(goConf , graph_BP);
        GraphLoaderGeneric.populate(goConf , graph_CC);
        GraphLoaderGeneric.populate(goConf , graph_MF);

        GraphLoaderGeneric.populate(annotConf, graph_BP);
        GraphLoaderGeneric.populate(annotConf, graph_CC);
        GraphLoaderGeneric.populate(annotConf, graph_MF);

        URI bpGOTerm = factory.getURI("http://go/0008150");
        GAction reduction_bp = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_bp.addParameter("root_uri", bpGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_bp, graph_BP);

        URI ccGOTerm = factory.getURI("http://go/0005575");
        GAction reduction_cc = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_cc.addParameter("root_uri", ccGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_cc, graph_CC);

        URI mfGOTerm = factory.getURI("http://go/0003674");
        GAction reduction_mf = new GAction(GActionType.VERTICES_REDUCTION);
        reduction_mf.addParameter("root_uri", mfGOTerm.stringValue());
        GraphActionExecutor.applyAction(factory, reduction_mf, graph_MF);

        int i = 0;
        for (String dataset_filename : dataset_files){
            ArrayList<String> pair_prots = get_proteins_dataset(dataset_filename);
            semantic_measures_2prots(graph_BP , graph_CC , graph_MF , factory , pair_prots , SSM_files[i]);
            i++;
        }


        t.stop();
        t.elapsedTime();
    }

    private ArrayList<String> get_proteins_dataset(String dataset_filename) throws  IOException{

        FileInputStream file_dataset = new FileInputStream(dataset_filename);
        BufferedReader br = new BufferedReader(new InputStreamReader(file_dataset));

        ArrayList<String> pairs_prots = new ArrayList<>();
        String strLine;

        // Read file line by line
        while ((strLine = br.readLine()) != null) {
            strLine = strLine.substring(0 , strLine.length()-1);
            pairs_prots.add(strLine);
        }

        file_dataset.close();
        return pairs_prots;


    }
    private void semantic_measures_2prots (G graph_BP, G graph_CC, G graph_MF, URIFactory factory, ArrayList<String> pairs_prots , String SSM_file) throws SLIB_Ex_Critic , FileNotFoundException{

        ICconf ic_Seco =  new IC_Conf_Topo("Seco" , SMConstants.FLAG_ICI_SECO_2004);
        ICconf ic_Resnik = new IC_Conf_Corpus("resnik" , SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED);

        SMconf SimGIC_icSeco = new SMconf("gic" , SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icSeco.setICconf(ic_Seco);

        SMconf SimGIC_icResnik = new SMconf("gic" , SMConstants.FLAG_SIM_GROUPWISE_DAG_GIC);
        SimGIC_icResnik.setICconf(ic_Resnik);

        SMconf Resnik_icSeco = new SMconf("resnik" , SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icSeco.setICconf(ic_Seco);

        SMconf Resnik_icResnik = new SMconf("resnik" , SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995);
        Resnik_icResnik.setICconf(ic_Resnik);

        SMconf max = new SMconf("max" , SMConstants.FLAG_SIM_GROUPWISE_MAX);
        SMconf bma = new SMconf("bma"  , SMConstants.FLAG_SIM_GROUPWISE_BMA);

        SM_Engine engine_bp = new SM_Engine(graph_BP);
        SM_Engine engine_cc = new SM_Engine(graph_CC);
        SM_Engine engine_mf = new SM_Engine(graph_MF);

        double sim_BP_SimGIC_icSeco, sim_CC_SimGIC_icSeco,  sim_MF_SimGIC_icSeco, sim_Avg_SimGIC_icSeco, sim_Max_SimGIC_icSeco;
        double sim_BP_ResnikMax_icSeco, sim_CC_ResnikMax_icSeco, sim_MF_ResnikMax_icSeco, sim_Avg_ResnikMax_icSeco, sim_Max_ResnikMax_icSeco;
        double sim_BP_ResnikBMA_icSeco, sim_CC_ResnikBMA_icSeco, sim_MF_ResnikBMA_icSeco, sim_Avg_ResnikBMA_icSeco, sim_Max_ResnikBMA_icSeco;

        double sim_BP_SimGIC_icResnik, sim_CC_SimGIC_icResnik,  sim_MF_SimGIC_icResnik, sim_Avg_SimGIC_icResnik, sim_Max_SimGIC_icResnik;
        double sim_BP_ResnikMax_icResnik, sim_CC_ResnikMax_icResnik, sim_MF_ResnikMax_icResnik, sim_Avg_ResnikMax_icResnik, sim_Max_ResnikMax_icResnik;
        double sim_BP_ResnikBMA_icResnik, sim_CC_ResnikBMA_icResnik, sim_MF_ResnikBMA_icResnik, sim_Avg_ResnikBMA_icResnik, sim_Max_ResnikBMA_icResnik;

        PrintWriter file = new PrintWriter(SSM_file);
        file.print("prot1   prot2   sim_BP_SimGIC_icSeco   sim_CC_SimGIC_icSeco   sim_MF_SimGIC_icSeco   sim_Avg_SimGIC_icSeco  sim_Max_SimGIC_icSeco  sim_BP_ResnikMax_icSeco    sim_CC_ResnikMax_icSeco    sim_MF_ResnikMax_icSeco    sim_Avg_ResnikMax_icSeco   sim_Max_ResnikMax_icSeco   sim_BP_ResnikBMA_icSeco    sim_CC_ResnikBMA_icSeco    sim_MF_ResnikBMA_icSeco    sim_Avg_ResnikBMA_icSeco   sim_Max_ResnikBMA_icSeco  sim_BP_SimGIC_icResnik   sim_CC_SimGIC_icResnik   sim_MF_SimGIC_icResnik   sim_Avg_SimGIC_icResnik  sim_Max_SimGIC_icResnik  sim_BP_ResnikMax_icResnik    sim_CC_ResnikMax_icResnik    sim_MF_ResnikMax_icResnik    sim_Avg_ResnikMax_icResnik   sim_Max_ResnikMax_icResnik   sim_BP_ResnikBMA_icResnik    sim_CC_ResnikBMA_icResnik    sim_MF_ResnikBMA_icResnik    sim_Avg_ResnikBMA_icResnik   sim_Max_ResnikBMA_icResnik" + "\n");


        for (String pair : pairs_prots){
            ArrayList<String> proteins = new ArrayList<String>(Arrays.asList(pair.split("\t")));
            String uri_prot1 = "http://go/" + proteins.get(0);
            String uri_prot2 = "http://go/" + proteins.get(1);

            URI instance1 = factory.getURI(uri_prot1);
            URI instance2 = factory.getURI(uri_prot2);

            if (((graph_BP.containsVertex(instance1))||(graph_CC.containsVertex(instance1))||((graph_MF.containsVertex(instance1))))&&
                    ((graph_BP.containsVertex(instance2))||(graph_CC.containsVertex(instance2))||(graph_MF.containsVertex(instance2)))) {
                InstancesAccessor gene_instance1_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance1_BP = gene_instance1_acessor_bp.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance1_CC = gene_instance1_acessor_cc.getDirectClass(instance1);

                InstancesAccessor gene_instance1_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance1_MF = gene_instance1_acessor_mf.getDirectClass(instance1);


                InstancesAccessor gene_instance2_acessor_bp = new InstanceAccessor_RDF_TYPE(graph_BP);
                Set<URI> annotations_instance2_BP = gene_instance2_acessor_bp.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_cc = new InstanceAccessor_RDF_TYPE(graph_CC);
                Set<URI> annotations_instance2_CC = gene_instance2_acessor_cc.getDirectClass(instance2);

                InstancesAccessor gene_instance2_acessor_mf = new InstanceAccessor_RDF_TYPE(graph_MF);
                Set<URI> annotations_instance2_MF = gene_instance2_acessor_mf.getDirectClass(instance2);

                if (instance1.equals(instance2)){
                    sim_BP_SimGIC_icSeco = sim_CC_SimGIC_icSeco = sim_MF_SimGIC_icSeco = 1;
                    sim_BP_ResnikMax_icSeco = sim_CC_ResnikMax_icSeco = sim_MF_ResnikMax_icSeco = 1;
                    sim_BP_ResnikBMA_icSeco = sim_CC_ResnikBMA_icSeco = sim_MF_ResnikBMA_icSeco = 1;

                    sim_BP_SimGIC_icResnik = sim_CC_SimGIC_icResnik = sim_MF_SimGIC_icResnik = 1;
                    sim_BP_ResnikMax_icResnik = sim_CC_ResnikMax_icResnik = sim_MF_ResnikMax_icResnik = 1;
                    sim_BP_ResnikBMA_icResnik = sim_CC_ResnikBMA_icResnik = sim_MF_ResnikBMA_icResnik = 1;


                } else {
                    if (annotations_instance1_BP.isEmpty() || annotations_instance2_BP.isEmpty()){
                        sim_BP_SimGIC_icSeco = sim_BP_ResnikMax_icSeco = sim_BP_ResnikBMA_icSeco = 0;
                        sim_BP_SimGIC_icResnik = sim_BP_ResnikMax_icResnik = sim_BP_ResnikBMA_icResnik = 0;

                    } else {
                        sim_BP_SimGIC_icSeco = engine_bp.compare(SimGIC_icSeco, annotations_instance1_BP, annotations_instance2_BP);
                        sim_BP_ResnikMax_icSeco = engine_bp.compare(max, Resnik_icSeco ,  annotations_instance1_BP, annotations_instance2_BP);
                        sim_BP_ResnikBMA_icSeco = engine_bp.compare(bma , Resnik_icSeco , annotations_instance1_BP, annotations_instance2_BP);

                        sim_BP_SimGIC_icResnik = engine_bp.compare(SimGIC_icResnik, annotations_instance1_BP, annotations_instance2_BP);
                        sim_BP_ResnikMax_icResnik = engine_bp.compare(max, Resnik_icResnik ,  annotations_instance1_BP, annotations_instance2_BP);
                        sim_BP_ResnikBMA_icResnik = engine_bp.compare(bma , Resnik_icResnik , annotations_instance1_BP, annotations_instance2_BP);

                    }


                    if (annotations_instance1_CC.isEmpty() || annotations_instance2_CC.isEmpty()){
                        sim_CC_SimGIC_icSeco = sim_CC_ResnikMax_icSeco = sim_CC_ResnikBMA_icSeco = 0;
                        sim_CC_SimGIC_icResnik = sim_CC_ResnikMax_icResnik = sim_CC_ResnikBMA_icResnik = 0;
                    } else {
                        sim_CC_SimGIC_icSeco = engine_cc.compare(SimGIC_icSeco, annotations_instance1_CC, annotations_instance2_CC);
                        sim_CC_ResnikMax_icSeco = engine_cc.compare(max , Resnik_icSeco, annotations_instance1_CC, annotations_instance2_CC);
                        sim_CC_ResnikBMA_icSeco = engine_cc.compare(bma , Resnik_icSeco, annotations_instance1_CC, annotations_instance2_CC);

                        sim_CC_SimGIC_icResnik = engine_cc.compare(SimGIC_icResnik, annotations_instance1_CC, annotations_instance2_CC);
                        sim_CC_ResnikMax_icResnik = engine_cc.compare(max , Resnik_icResnik, annotations_instance1_CC, annotations_instance2_CC);
                        sim_CC_ResnikBMA_icResnik = engine_cc.compare(bma , Resnik_icResnik, annotations_instance1_CC, annotations_instance2_CC);

                    }


                    if (annotations_instance1_MF.isEmpty() || annotations_instance2_MF.isEmpty()){
                        sim_MF_SimGIC_icSeco = sim_MF_ResnikMax_icSeco = sim_MF_ResnikBMA_icSeco = 0;
                        sim_MF_SimGIC_icResnik = sim_MF_ResnikMax_icResnik = sim_MF_ResnikBMA_icResnik = 0;
                    } else {
                        sim_MF_SimGIC_icSeco = engine_mf.compare(SimGIC_icSeco, annotations_instance1_MF, annotations_instance2_MF);
                        sim_MF_ResnikMax_icSeco = engine_mf.compare(max, Resnik_icSeco , annotations_instance1_MF, annotations_instance2_MF);
                        sim_MF_ResnikBMA_icSeco = engine_mf.compare(bma , Resnik_icSeco , annotations_instance1_MF, annotations_instance2_MF);

                        sim_MF_SimGIC_icResnik = engine_mf.compare(SimGIC_icResnik, annotations_instance1_MF, annotations_instance2_MF);
                        sim_MF_ResnikMax_icResnik = engine_mf.compare(max, Resnik_icResnik , annotations_instance1_MF, annotations_instance2_MF);
                        sim_MF_ResnikBMA_icResnik = engine_mf.compare(bma , Resnik_icResnik , annotations_instance1_MF, annotations_instance2_MF);
                    }
                }

                sim_Avg_SimGIC_icSeco = (sim_BP_SimGIC_icSeco + sim_CC_SimGIC_icSeco + sim_MF_SimGIC_icSeco) / 3;
                sim_Avg_ResnikMax_icSeco = (sim_BP_ResnikMax_icSeco + sim_CC_ResnikMax_icSeco + sim_MF_ResnikMax_icSeco) /3;
                sim_Avg_ResnikBMA_icSeco = (sim_BP_ResnikBMA_icSeco + sim_CC_ResnikBMA_icSeco + sim_MF_ResnikBMA_icSeco) /3;

                sim_Avg_SimGIC_icResnik = (sim_BP_SimGIC_icResnik + sim_CC_SimGIC_icResnik + sim_MF_SimGIC_icResnik) / 3;
                sim_Avg_ResnikMax_icResnik = (sim_BP_ResnikMax_icResnik + sim_CC_ResnikMax_icResnik + sim_MF_ResnikMax_icResnik) /3;
                sim_Avg_ResnikBMA_icResnik = (sim_BP_ResnikBMA_icResnik + sim_CC_ResnikBMA_icResnik + sim_MF_ResnikBMA_icResnik) /3;

                sim_Max_SimGIC_icSeco = Math.max(Math.max(sim_BP_SimGIC_icSeco , sim_CC_SimGIC_icSeco) , sim_MF_SimGIC_icSeco);
                sim_Max_ResnikMax_icSeco= Math.max(Math.max(sim_BP_ResnikMax_icSeco , sim_CC_ResnikMax_icSeco) , sim_MF_ResnikMax_icSeco);
                sim_Max_ResnikBMA_icSeco = Math.max(Math.max(sim_BP_ResnikBMA_icSeco , sim_CC_ResnikBMA_icSeco) , sim_MF_ResnikBMA_icSeco);

                sim_Max_SimGIC_icResnik = Math.max(Math.max(sim_BP_SimGIC_icResnik , sim_CC_SimGIC_icResnik) , sim_MF_SimGIC_icResnik);
                sim_Max_ResnikMax_icResnik= Math.max(Math.max(sim_BP_ResnikMax_icResnik , sim_CC_ResnikMax_icResnik) , sim_MF_ResnikMax_icResnik);
                sim_Max_ResnikBMA_icResnik = Math.max(Math.max(sim_BP_ResnikBMA_icResnik , sim_CC_ResnikBMA_icResnik) , sim_MF_ResnikBMA_icResnik);

                file.print(instance1 + "\t" + instance2 + "\t" + sim_BP_SimGIC_icSeco + "\t" + sim_CC_SimGIC_icSeco + "\t" + sim_MF_SimGIC_icSeco +
                        "\t" + sim_Avg_SimGIC_icSeco + "\t" +  sim_Max_SimGIC_icSeco + "\t" +
                        sim_BP_ResnikMax_icSeco + "\t" + sim_CC_ResnikMax_icSeco + "\t" + sim_MF_ResnikMax_icSeco + "\t" +
                        sim_Avg_ResnikMax_icSeco + "\t" + sim_Max_ResnikMax_icSeco + "\t" +
                        sim_BP_ResnikBMA_icSeco + "\t" + sim_CC_ResnikBMA_icSeco + "\t" + sim_MF_ResnikBMA_icSeco + "\t" +
                        sim_Avg_ResnikBMA_icSeco + "\t" + sim_Max_ResnikBMA_icSeco + "\t" +
                        sim_BP_SimGIC_icResnik + "\t" + sim_CC_SimGIC_icResnik + "\t" + sim_MF_SimGIC_icResnik + "\t" +
                        sim_Avg_SimGIC_icResnik + "\t" +  sim_Max_SimGIC_icResnik + "\t" +
                        sim_BP_ResnikMax_icResnik + "\t" + sim_CC_ResnikMax_icResnik + "\t" + sim_MF_ResnikMax_icResnik + "\t" +
                        sim_Avg_ResnikMax_icResnik + "\t" + sim_Max_ResnikMax_icResnik + "\t" +
                        sim_BP_ResnikBMA_icResnik + "\t" + sim_CC_ResnikBMA_icResnik + "\t" + sim_MF_ResnikBMA_icResnik + "\t" +
                        sim_Avg_ResnikBMA_icResnik + "\t" + sim_Max_ResnikBMA_icResnik + "\n");
               file.flush();
            }
        }

        file.close();
    }

}


public class Run_SS_calculation {

    public static void main(String[] args) throws Exception {

        // The implementation of SML requires a annotation file in GAF 2.0. Since the most recent GO annotation file is in GAF 2.1 format, it was converted to the older format specifications.
        Convert_GAF_versions human_annot = new Convert_GAF_versions("Data/GeneOntology_data/goa_human.gaf", "Data/Processed_GeneOntology_data/goa_human_20.gaf");
        human_annot.run();

        // Calculate the SS for human datasets
        Calculate_sim_prot human_datasets = new Calculate_sim_prot("Data/GeneOntology_data/go.obo", "Data/Processed_GeneOntology_data/goa_human_20.gaf",
                new String[]{"SS_Calculation/SS_files/SS_DIP_HS.txt",
                        "SS_Calculation/SS_files/SS_STRING_HS.txt",
                        "SS_Calculation/SS_files/SS_GRIDHPRD_bal_HS.txt",
                        "SS_Calculation/SS_files/SS_GRIDHPRD_unbal_HS.txt"},
                new String[]{"Data/Processed_PPIdatasets/DIP_HS/DIP_HS.txt",
                        "Data/Processed_PPIdatasets/STRING_HS/STRING_HS.txt",
                        "Data/Processed_PPIdatasets/GRIDHPRD_bal_HS/GRIDHPRD_bal_HS.txt",
                        "Data/Processed_PPIdatasets/GRIDHPRD_unbal_HS/GRIDHPRD_unbal_HS.txt"});
        human_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for human completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for E.coli dataset
        Convert_GAF_versions ecoli_annot = new Convert_GAF_versions("Data/GeneOntology_data/ecocyc.gaf", "Data/Processed_GeneOntology_data/goa_ecoli_20.gaf");
        ecoli_annot.run();
        Calculate_sim_prot ecoli_datasets = new Calculate_sim_prot("Data/GeneOntology_data/go.obo", "Data/Processed_GeneOntology_data/goa_ecoli_20.gaf",
                new String[]{"SS_Calculation/SS_files/SS_STRING_EC.txt"},
                new String[]{"Data/Processed_PPIdatasets/STRING_EC/STRING_EC.txt"});
        ecoli_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for E. coli completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for fly dataset
        Convert_GAF_versions fly_annot = new Convert_GAF_versions("Data/GeneOntology_data/goa_fly.gaf","Data/Processed_GeneOntology_data/goa_fly_20.gaf");
        fly_annot.run();
        Calculate_sim_prot fly_datasets = new Calculate_sim_prot("Data/GeneOntology_data/go.obo", "Data/Processed_GeneOntology_data/goa_fly_20.gaf",
                new String[]{"SS_Calculation/SS_files/SS_STRING_DM.txt"},
                new String[]{"Data/Processed_PPIdatasets/STRING_DM/STRING_DM.txt"});
        fly_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for D. melanogaster completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for yeast datasets
        Convert_GAF_versions yeast_annot = new Convert_GAF_versions("Data/GeneOntology_data/goa_yeast.gaf", "Data/Processed_GeneOntology_data/goa_yeast_20.gaf");
        yeast_annot.run();
        Calculate_sim_prot yeast_datasets = new Calculate_sim_prot("Data/GeneOntology_data/go.obo", "Data/Processed_GeneOntology_data/goa_yeast_20.gaf",
                new String[]{"SS_Calculation/SS_files/SS_STRING_SC.txt",
                        "SS_Calculation/SS_files/SS_DIPMIPS_SC.txt",
                        "SS_Calculation/SS_files/SS_BIND_SC.txt"},
                new String[]{"Data/Processed_PPIdatasets/STRING_SC/STRING_SC.txt",
                        "Data/Processed_PPIdatasets/DIPMIPS_SC/DIPMIPS_SC.txt",
                        "Data/Processed_PPIdatasets/BIND_SC/BIND_SC.txt"});
        yeast_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for S. cerevisiae completed.");
        System.out.println("---------------------------------------------------------------");

/*        Convert_GAF_versions human_annot = new Convert_GAF_versions("Data/Processed_kgsimDatasets/goa_human.gaf", "Data/Processed_kgsimDatasets/goa_human_new.gaf");
        human_annot.run();
        Calculate_sim_prot human_datasets = new Calculate_sim_prot("Data/Processed_kgsimDatasets/go-basic.obo", "Data/Processed_kgsimDatasets/goa_human_new.gaf",
                new String[]{"SS_Calculation/SS_files/SS_MF_HS1.txt",
                        "SS_Calculation/SS_files/SS_MF_HS3.txt",
                        "SS_Calculation/SS_files/SS_PPI_HS1.txt",
                        "SS_Calculation/SS_files/SS_PPI_HS3.txt"},
                new String[]{"Data/Processed_kgsimDatasets/MF_HS1/SEQ/MF_HS1.txt",
                        "Data/Processed_kgsimDatasets/MF_HS3/SEQ/MF_HS3.txt",
                        "Data/Processed_kgsimDatasets/PPI_HS1/SEQ/PPI_HS1.txt",
                        "Data/Processed_kgsimDatasets/PPI_HS3/SEQ/PPI_HS3.txt"});
        human_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for human completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for E.coli dataset of kgsim benchmarcks
        Convert_GAF_versions ecoli_annot = new Convert_GAF_versions("Data/Processed_kgsimDatasets/goa_ecoli.gaf", "Data/Processed_kgsimDatasets/goa_ecoli_new.gaf");
        ecoli_annot.run();
        Calculate_sim_prot ecoli_datasets = new Calculate_sim_prot("Data/Processed_kgsimDatasets/go-basic.obo", "Data/Processed_kgsimDatasets/goa_ecoli_new.gaf",
                new String[]{"SS_Calculation/SS_files/SS_MF_EC1.txt",
                        "SS_Calculation/SS_files/SS_MF_EC3.txt",
                        "SS_Calculation/SS_files/SS_PPI_EC1.txt",
                        "SS_Calculation/SS_files/SS_PPI_EC3.txt"},
                new String[]{"Data/Processed_kgsimDatasets/MF_EC1/SEQ/MF_EC1.txt",
                        "Data/Processed_kgsimDatasets/MF_EC3/SEQ/MF_EC3.txt",
                        "Data/Processed_kgsimDatasets/PPI_EC1/SEQ/PPI_EC1.txt",
                        "Data/Processed_kgsimDatasets/PPI_EC3/SEQ/PPI_EC3.txt"});
        ecoli_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for E. coli completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for fly dataset of kgsim benchmarcks
        Convert_GAF_versions fly_annot = new Convert_GAF_versions("Data/Processed_kgsimDatasets/goa_fly.gaf","Data/Processed_kgsimDatasets/goa_fly_new.gaf");
        fly_annot.run();
        Calculate_sim_prot fly_datasets = new Calculate_sim_prot("Data/Processed_kgsimDatasets/go-basic.obo", "Data/Processed_kgsimDatasets/goa_fly_new.gaf",
                new String[]{"SS_Calculation/SS_files/SS_MF_DM1.txt",
                        "SS_Calculation/SS_files/SS_MF_DM3.txt",
                        "SS_Calculation/SS_files/SS_PPI_DM1.txt",
                        "SS_Calculation/SS_files/SS_PPI_DM3.txt"},
                new String[]{"Data/Processed_kgsimDatasets/MF_DM1/SEQ/MF_DM1.txt",
                        "Data/Processed_kgsimDatasets/MF_DM3/SEQ/MF_DM3.txt",
                        "Data/Processed_kgsimDatasets/PPI_DM1/SEQ/PPI_DM1.txt",
                        "Data/Processed_kgsimDatasets/PPI_DM3/SEQ/PPI_DM3.txt"});
        fly_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for D. melanogaster completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for yeast datasets of kgsim benchmarcks
        Convert_GAF_versions yeast_annot = new Convert_GAF_versions("Data/Processed_kgsimDatasets/goa_yeast.gaf", "Data/Processed_kgsimDatasets/goa_yeast_new.gaf");
        yeast_annot.run();
        Calculate_sim_prot yeast_datasets = new Calculate_sim_prot("Data/Processed_kgsimDatasets/go-basic.obo", "Data/Processed_kgsimDatasets/goa_yeast_new.gaf",
                new String[]{"SS_Calculation/SS_files/SS_MF_SC1.txt",
                        "SS_Calculation/SS_files/SS_MF_SC3.txt",
                        "SS_Calculation/SS_files/SS_PPI_SC1.txt",
                        "SS_Calculation/SS_files/SS_PPI_SC3.txt"},
                new String[]{"Data/Processed_kgsimDatasets/MF_SC1/SEQ/MF_SC1.txt",
                        "Data/Processed_kgsimDatasets/MF_SC3/SEQ/MF_SC3.txt",
                        "Data/Processed_kgsimDatasets/PPI_SC1/SEQ/PPI_SC1.txt",
                        "Data/Processed_kgsimDatasets/PPI_SC3/SEQ/PPI_SC3.txt"});
        yeast_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for S. cerevisiae completed.");
        System.out.println("---------------------------------------------------------------");

        // Calculate the SS for 4SPECIES datasets of kgsim benchmarcks
        Convert_GAF_versions all_annot = new Convert_GAF_versions("Data/Processed_kgsimDatasets/goa_4species.gaf", "Data/Processed_kgsimDatasets/goa_4species_new.gaf");
        all_annot.run();
        Calculate_sim_prot all_datasets = new Calculate_sim_prot("Data/Processed_kgsimDatasets/go-basic.obo", "Data/Processed_kgsimDatasets/goa_4species_new.gaf",
                new String[]{"SS_Calculation/SS_files/SS_MF_ALL1.txt",
                        "SS_Calculation/SS_files/SS_MF_ALL3.txt",
                        "SS_Calculation/SS_files/SS_PPI_ALL1.txt",
                        "SS_Calculation/SS_files/SS_PPI_ALL3.txt"},
                new String[]{"Data/Processed_kgsimDatasets/MF_ALL1/SEQ/MF_ALL1.txt",
                        "Data/Processed_kgsimDatasets/MF_ALL3/SEQ/MF_ALL3.txt",
                        "Data/Processed_kgsimDatasets/PPI_ALL1/SEQ/PPI_ALL1.txt",
                        "Data/Processed_kgsimDatasets/PPI_ALL3/SEQ/PPI_ALL3.txt"});
        all_datasets.run();
        System.out.println("---------------------------------------------------------------");
        System.out.println("SSM for 4SPECIES completed.");
        System.out.println("---------------------------------------------------------------");*/

    }

}
