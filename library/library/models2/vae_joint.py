



class JointVae(nn.Module):
    """
    """
    def __init__(self, 
                latent_min_capacity: float = 0.,
                latent_max_capactiy: float = 25.,
                latent_gamma: float = 30,
                latent_num_iter: int = 25000,
                categorical_min_capacity: float = 0., 
                categorical_max_capacity: float = 25.,
                categorical_gamma: float = 30.,
                categorical_num_iter: int = 25000,
                temperature: float = 0.5,
                anneal_rate: float = 3e-5,
                anneal_interval: int = 100,
                alpha: float = 30.,
                **kwargs):
        super(JointVae, self).__init__(
            **kwargs
        )

        self.latent_dim = self.img_encoder.latent_dim
        self.hidden_dim = self.img_encoder.enc_hidden_dims
        self.output_dim = self.img_encoder.enc_output_dim

        self.mu = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.latent_dim)
        self.logvar = nn.Linear(self.hidden_dim[-1] * self.ouput_dim, self.latent_dim)
        self.q = nn.Linear(self.hidden_dim[-1] * self.output_dim, self.categorical_dim)

        self.temperature = temperature
        self.min_temperature = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval 
        self.alpha = alpha 

        self.latent_min_capacity = latent_min_capacity
    
    def _reparameterization(self, h_enc, epsilon=1e-7):
        """
        """
        mu = self.mu(h_enc)
        logvar = self.logvar(h_enc)
        q = self.q(h_enc)
        q = q.view(-1, self.categorical_dim)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        self.loss_item['mu'] = mu
        self.loss_item['logvar'] = logvar

        z = eps * std + mu 

        u = torch.randn_like(q)
        g = - torch.log(-torch.log(u + epsilon) + epsilon)

        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dim)

        return torch.cat([z, s], dim=1)
    
    def _loss_function(self, image=None, text=None, recon_image=None,
                    recon_text=None, mu=None, logvar=None, *args, **kwargs):

        
