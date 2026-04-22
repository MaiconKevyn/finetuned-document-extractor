# 🎓 Fundamentos Técnicos: Do Modelo ao Kubernetes

Este documento explica os conceitos fundamentais e o passo a passo técnico realizado a partir da **Fase 4**, conectando a teoria da engenharia de software com a prática de MLOps.

---

## 1. O que é Kubernetes (K8s)?

Imagine que você tem um restaurante (seu projeto).
- **Docker** é o "Contêiner": É a marmita pronta, com todos os ingredientes e a receita dentro. Ela tem o mesmo sabor em qualquer lugar.
- **Kubernetes** é o "Gerente do Restaurante": Ele não cozinha, mas decide quantas cozinheiras (contêineres) precisam estar trabalhando, o que fazer se uma passar mal (o pod cair) e como levar o prato até o cliente (o service).

### Conceitos Chave:
*   **Pod:** É a menor unidade no K8s. Pense nele como uma "caixa" que envolve o seu contêiner Docker. Um Pod pode ser destruído e recriado a qualquer momento.
*   **Node (Nó):** É a máquina física ou virtual onde os Pods moram. No seu caso, o seu PC é o único Nó do cluster.
*   **Deployment:** É o "chefe" dos Pods. Você diz a ele: "Quero que sempre existam 2 Pods da API DocTune rodando". Se um travar, o Deployment percebe e sobe outro instantaneamente.
*   **Service:** Como os Pods mudam de IP toda vez que morrem/nascem, o Service funciona como um "IP Fixo" (ou nome de rede) que sempre aponta para os Pods que estiverem vivos.

---

## 2. Passo a Passo: Fase 4 (MLOps & API)

Nesta fase, transformamos um script de pesquisa em um **produto de software**.

### O Desafio da VRAM e o Padrão Singleton
Modelos de IA são gigantes. Se cada vez que um usuário fizesse uma pergunta a API carregasse o modelo, a memória da sua GPU (8GB) estouraria em segundos.
- **O que fizemos:** Criamos um **Singleton**. O modelo é carregado uma única vez quando a API sobe. Ele fica "estacionado" na VRAM, pronto para receber ordens.

### Docker: A Imutabilidade
O Docker resolve o problema do "na minha máquina funciona". No `Dockerfile`, usamos uma imagem base da NVIDIA que já vem com os drivers de GPU (CUDA) instalados. Isso garante que o ambiente de execução seja idêntico, seja no seu PC ou em um servidor na nuvem.

---

## 3. Passo a Passo: Fase 5 (O Salto para o Kubernetes)

Aqui entramos no mundo da infraestrutura de alta escala.

### K3s: Kubernetes de Baixo Consumo
Usamos o **K3s** em vez do Kubernetes padrão porque ele é otimizado para hardware limitado. Ele consome muito menos RAM, deixando o resto para o seu modelo de IA.

### O Mistério do NVIDIA Device Plugin
Por padrão, o Kubernetes só entende CPU e RAM. Ele ignora a existência da sua placa de vídeo.
- **O que fizemos:** Instalamos o **NVIDIA Device Plugin**. Ele atua como um "tradutor". Ele varre o seu PC, descobre a RTX 2070 e avisa ao Kubernetes: "Ei, eu tenho 1 unidade de GPU disponível aqui!".
- É por isso que no arquivo `k8s/deployment.yaml` pudemos escrever `nvidia.com/gpu: 1`.

### Probes: Entendendo o "Tempo da IA"
Um servidor comum (como um site simples) sobe em 1 segundo. Um modelo de IA leva 40 segundos para mover 3GB de pesos do disco para a VRAM.
- **Readiness Probe:** Diz ao Kubernetes: "Não mande tráfego para este Pod ainda! Ele está carregando os pesos da IA". Sem isso, o usuário receberia um erro enquanto o modelo carrega.
- **Liveness Probe:** Monitora se a API travou. Se o modelo "congelar" a GPU, o Kubernetes mata o Pod e sobe um novo automaticamente (Self-healing).

---

## 4. Fluxo de Deploy Industrial (Resumo)

1.  **Código:** Escrevemos a API (FastAPI).
2.  **Pacote:** Criamos a imagem (Docker).
3.  **Injeção:** Movemos a imagem para o cluster (`ctr images import`).
4.  **Manifesto:** Declaramos em YAML o que queremos (1 Pod, 1 GPU, Health Checks).
5.  **Orquestração:** O Kubernetes lê o YAML, reserva a GPU na sua RTX 2070, sobe o contêiner e monitora a saúde dele.

---

## 💡 Por que isso é "Nível Indústria"?
Empresas como Netflix e Uber não rodam scripts. Elas rodam **serviços orquestrados**. O que você construiu segue exatamente esse padrão: se a demanda aumentar, bastaria mudar `replicas: 1` para `replicas: 10` em um cluster maior, e o Kubernetes cuidaria de distribuir o modelo em 10 GPUs diferentes sem você mudar uma linha de código.
